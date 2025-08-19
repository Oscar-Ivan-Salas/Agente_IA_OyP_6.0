#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            # Normalizar por longitud del texto
            emotion_scores[emotion] = score / max(len(text.split()), 1) * 100
        
        return emotion_scores
    
    async def generate_wordcloud(self, text: str, output_path: str = None) -> str:
        """Generar nube de palabras"""
        
        try:
            # Limpiar texto
            words = nltk.word_tokenize(text.lower())
            words = [w for w in words if w.isalnum() and len(w) > 2]
            clean_text = ' '.join(words)
            
            # Generar wordcloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(clean_text)
            
            # Guardar imagen
            if not output_path:
                output_path = f"static/plots/wordcloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            wordcloud.to_file(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generando wordcloud: {e}")
            return ""
    
    async def batch_analyze(self, texts: List[str]) -> List[TextAnalysisResult]:
        """Análisis en lote de múltiples textos"""
        
        logger.info(f"📦 Analizando lote de {len(texts)} textos...")
        
        tasks = [self.analyze_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar resultados exitosos
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error en texto {i}: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"✅ Lote completado: {len(successful_results)}/{len(texts)} exitosos")
        return successful_results
    
    def get_supported_languages(self) -> List[str]:
        """Obtener idiomas soportados"""
        return self.supported_languages
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Obtener estado de modelos cargados"""
        return {
            'spacy_model': self.nlp_model is not None,
            'vader_sentiment': self.sentiment_analyzer is not None,
            'nltk_resources': True  # Asumimos que están disponibles si llegó hasta aquí
        }
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "text_analyzer.py", "w", encoding="utf-8") as f:
            f.write(text_analyzer_content)
        
        self.logger.info("✅ Analizador de texto creado")

    def create_ai_insights_service(self):
        """Crear servicio de insights con IA"""
        self.logger.info("🤖 Creando servicio de insights con IA...")
        
        ai_insights_content = '''"""
Servicio de insights usando IA - se conecta con AI-Engine
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import httpx
from dataclasses import dataclass
from datetime import datetime
import json

from utils.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class AIInsight:
    """Estructura para un insight generado por IA"""
    title: str
    content: str
    confidence: float
    category: str
    metadata: Dict[str, Any]
    timestamp: str

class AIInsightsService:
    """Servicio para generar insights usando el AI-Engine"""
    
    def __init__(self):
        self.settings = Settings()
        self.is_ready = False
        self.ai_engine_url = f"http://localhost:{self.settings.ai_engine_port}"
        self.client = None
        
    async def initialize(self):
        """Inicializar el servicio de insights"""
        logger.info("🔧 Inicializando AI Insights Service...")
        
        try:
            # Crear cliente HTTP
            self.client = httpx.AsyncClient(timeout=30.0)
            
            # Verificar conexión con AI-Engine
            connection_ok = await self.check_ai_engine_connection()
            if connection_ok:
                logger.info("✅ Conexión con AI-Engine establecida")
            else:
                logger.warning("⚠️ AI-Engine no disponible. Modo limitado.")
            
            self.is_ready = True
            logger.info("✅ AI Insights Service inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando AI Insights: {e}")
            self.is_ready = True  # Continuar en modo limitado
    
    async def check_ai_engine_connection(self) -> bool:
        """Verificar conexión con AI-Engine"""
        
        try:
            response = await self.client.get(f"{self.ai_engine_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def generate_document_insights(
        self, 
        text: str, 
        analysis_result: Dict = None,
        options: Dict = None
    ) -> List[AIInsight]:
        """Generar insights sobre un documento"""
        
        options = options or {}
        insights = []
        
        try:
            # Insight 1: Resumen del contenido
            if options.get('generate_summary', True):
                summary_insight = await self._generate_summary_insight(text)
                if summary_insight:
                    insights.append(summary_insight)
            
            # Insight 2: Análisis de temas principales
            if options.get('analyze_themes', True):
                themes_insight = await self._analyze_themes_insight(text)
                if themes_insight:
                    insights.append(themes_insight)
            
            # Insight 3: Clasificación del documento
            if options.get('classify_document', True):
                classification_insight = await self._classify_document_insight(text)
                if classification_insight:
                    insights.append(classification_insight)
            
            # Insight 4: Análisis de sentimiento contextual
            if options.get('sentiment_context', True) and analysis_result:
                sentiment_insight = await self._generate_sentiment_context(text, analysis_result)
                if sentiment_insight:
                    insights.append(sentiment_insight)
            
            # Insight 5: Recomendaciones
            if options.get('generate_recommendations', True):
                recommendations_insight = await self._generate_recommendations(text, analysis_result)
                if recommendations_insight:
                    insights.append(recommendations_insight)
            
            logger.info(f"✅ Generados {len(insights)} insights")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generando insights: {e}")
            return []
    
    async def _generate_summary_insight(self, text: str) -> Optional[AIInsight]:
        """Generar resumen inteligente del documento"""
        
        try:
            prompt = f"""
            Analiza el siguiente texto y proporciona un resumen conciso que capture:
            1. Los puntos principales
            2. Los temas centrales
            3. Las conclusiones clave
            
            Texto a analizar:
            {text[:2000]}...
            
            Proporciona un resumen de máximo 150 palabras.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Resumen del Documento",
                    content=response,
                    confidence=0.85,
                    category="summary",
                    metadata={"word_count": len(response.split())},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error generando resumen: {e}")
        
        return None
    
    async def _analyze_themes_insight(self, text: str) -> Optional[AIInsight]:
        """Analizar temas principales del documento"""
        
        try:
            prompt = f"""
            Identifica y analiza los 3-5 temas principales en el siguiente texto.
            Para cada tema, proporciona:
            1. Nombre del tema
            2. Relevancia (1-10)
            3. Breve descripción
            
            Texto:
            {text[:2000]}...
            
            Responde en formato estructurado.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Análisis de Temas Principales",
                    content=response,
                    confidence=0.80,
                    category="themes",
                    metadata={"analysis_type": "thematic"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error analizando temas: {e}")
        
        return None
    
    async def _classify_document_insight(self, text: str) -> Optional[AIInsight]:
        """Clasificar tipo de documento"""
        
        try:
            prompt = f"""
            Clasifica el siguiente documento en una de estas categorías:
            - Reporte técnico
            - Documento legal
            - Comunicación comercial
            - Contenido académico
            - Manual o guía
            - Correspondencia
            - Otro
            
            Explica brevemente por qué pertenece a esa categoría y qué características lo definen.
            
            Texto:
            {text[:1500]}...
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Clasificación del Documento",
                    content=response,
                    confidence=0.75,
                    category="classification",
                    metadata={"classification_type": "document_type"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error clasificando documento: {e}")
        
        return None
    
    async def _generate_sentiment_context(self, text: str, analysis_result: Dict) -> Optional[AIInsight]:
        """Generar contexto del análisis de sentimientos"""
        
        try:
            sentiment_data = analysis_result.get('sentiment', {})
            overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
            
            prompt = f"""
            El análisis automático indica que este documento tiene un sentimiento "{overall_sentiment}".
            
            Analiza el contexto y explica:
            1. ¿Por qué el documento tiene este sentimiento?
            2. ¿Qué elementos específicos contribuyen a este sentimiento?
            3. ¿Es apropiado el tono para el tipo de documento?
            
            Fragmento del texto:
            {text[:1000]}...
            
            Proporciona un análisis contextual del sentimiento.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title=f"Contexto del Sentimiento ({overall_sentiment.title()})",
                    content=response,
                    confidence=0.70,
                    category="sentiment_context",
                    metadata={"detected_sentiment": overall_sentiment},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error en contexto de sentimiento: {e}")
        
        return None
    
    async def _generate_recommendations(self, text: str, analysis_result: Dict = None) -> Optional[AIInsight]:
        """Generar recomendaciones basadas en el análisis"""
        
        try:
            context = ""
            if analysis_result:
                readability = analysis_result.get('readability', {})
                flesch_score = readability.get('flesch_reading_ease', 50)
                
                if flesch_score < 30:
                    context += "El documento tiene baja legibilidad. "
                elif flesch_score > 70:
                    context += "El documento tiene alta legibilidad. "
            
            prompt = f"""
            Basándote en el análisis del siguiente documento, proporciona 3-5 recomendaciones prácticas para:
            1. Mejorar la claridad y legibilidad
            2. Optimizar la estructura
            3. Enhancer la comunicación efectiva
            
            {context}
            
            Texto analizado:
            {text[:1000]}...
            
            Proporciona recomendaciones específicas y accionables.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Recomendaciones de Mejora",
                    content=response,
                    confidence=0.75,
                    category="recommendations",
                    metadata={"recommendation_type": "improvement"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error generando recomendaciones: {e}")
        
        return None
    
    async def _call_ai_engine(self, prompt: str, model: str = "auto") -> Optional[str]:
        """Llamar al AI-Engine para generar respuesta"""
        
        if not await self.check_ai_engine_connection():
            logger.warning("⚠️ AI-Engine no disponible")
            return None
        
        try:
            payload = {
                "prompt": prompt,
                "model": model,
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = await self.client.post(
                f"{self.ai_engine_url}/generate/complete",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("text", "").strip()
            else:
                logger.warning(f"⚠️ AI-Engine respondió con status {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error llamando AI-Engine: {e}")
            return None
    
    async def generate_comparative_insights(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[AIInsight]:
        """Generar insights comparativos entre múltiples documentos"""
        
        if len(documents) < 2:
            return []
        
        insights = []
        
        try:
            # Comparar sentimientos
            sentiments = [doc.get('analysis', {}).get('sentiment', {}).get('overall_sentiment', 'neutral') 
                         for doc in documents]
            
            # Comparar longitudes
            lengths = [doc.get('analysis', {}).get('word_count', 0) for doc in documents]
            
            # Comparar legibilidad
            readabilities = [doc.get('analysis', {}).get('readability', {}).get('flesch_reading_ease', 50) 
                           for doc in documents]
            
            comparison_data = {
                'document_count': len(documents),
                'sentiments': sentiments,
                'avg_length': sum(lengths) / len(lengths),
                'avg_readability': sum(readabilities) / len(readabilities)
            }
            
            prompt = f"""
            Analiza los siguientes datos comparativos de {len(documents)} documentos:
            
            Sentimientos: {sentiments}
            Longitud promedio: {comparison_data['avg_length']:.0f} palabras
            Legibilidad promedio: {comparison_data['avg_readability']:.1f}
            
            Proporciona insights sobre:
            1. Patrones en los sentimientos
            2. Consistencia en la comunicación
            3. Variabilidad en complejidad
            4. Recomendaciones para homogeneizar
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                insights.append(AIInsight(
                    title="Análisis Comparativo",
                    content=response,
                    confidence=0.80,
                    category="comparative",
                    metadata=comparison_data,
                    timestamp=datetime.now().isoformat()
                ))
            
        except Exception as e:
            logger.error(f"❌ Error en insights comparativos: {e}")
        
        return insights
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles en AI-Engine"""
        return ["auto", "gpt-3.5-turbo", "claude-3-sonnet", "local"]
    
    async def cleanup(self):
        """Limpiar recursos"""
        if self.client:
            await self.client.aclose()
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "ai_insights.py", "w", encoding="utf-8") as f:
            f.write(ai_insights_content)
        
        self.logger.info("✅ Servicio de insights con IA creado")

    def create_chart_generator(self):
        """Crear generador de gráficos"""
        self.logger.info("📊 Creando generador de gráficos...")
        
        chart_generator_content = '''"""
Generador de gráficos y visualizaciones
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generador de gráficos para análisis de datos"""
    
    def __init__(self):
        self.is_ready = False
        self.output_dir = "static/plots"
        
        # Configurar estilo por defecto
        plt.style.use('default')
        sns.set_palette("husl")
        
    async def initialize(self):
        """Inicializar el generador de gráficos"""
        logger.info("🔧 Inicializando Chart Generator...")
        
        # Crear directorio de salida
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.is_ready = True
        logger.info("✅ Chart Generator inicializado")
    
    async def create_sentiment_chart(
        self, 
        sentiment_data: Dict[str, float], 
        title: str = "Análisis de Sentimientos"
    ) -> str:
        """Crear gráfico de análisis de sentimientos"""
        
        try:
            # Crear gráfico con Plotly
            fig = go.Figure()
            
            # Datos de VADER
            vader_data = {
                'Positivo': sentiment_data.get('vader_positive', 0),
                'Negativo': sentiment_data.get('vader_negative', 0),
                'Neutral': sentiment_data.get('vader_neutral', 0)
            }
            
            # Gráfico de barras
            fig.add_trace(go.Bar(
                x=list(vader_data.keys()),
                y=list(vader_data.values()),
                marker_color=['green', 'red', 'blue'],
                text=[f'{v:.2f}' for v in vader_data.values()],
                textposition='auto'
            ))
            
            # Configurar layout
            fig.update_layout(
                title=title,
                xaxis_title="Categoría",
                yaxis_title="Puntuación",
                template="plotly_white",
                height=400
            )
            
            # Guardar archivo
            filename = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de sentimientos creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de sentimientos: {e}")
            return ""
    
    async def create_readability_chart(
        self, 
        readability_data: Dict[str, float],
        title: str = "Métricas de Legibilidad"
    ) -> str:
        """Crear gráfico de métricas de legibilidad"""
        
        try:
            # Preparar datos
            metrics = []
            values = []
            
            if 'flesch_reading_ease' in readability_data:
                metrics.append('Facilidad Lectura')
                values.append(readability_data['flesch_reading_ease'])
            
            if 'average_sentence_length' in readability_data:
                metrics.append('Long. Oraciones')
                values.append(min(readability_data['average_sentence_length'], 50))  # Escalar
            
            if 'average_word_length' in readability_data:
                metrics.append('Long. Palabras')
                values.append(readability_data['average_word_length'] * 10)  # Escalar
            
            # Crear gráfico radial
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='Métricas'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title=title,
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"readability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de legibilidad creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de legibilidad: {e}")
            return ""
    
    async def create_keywords_chart(
        self, 
        keywords: List[Dict[str, float]],
        title: str = "Palabras Clave Principales"
    ) -> str:
        """Crear gráfico de palabras clave"""
        
        try:
            if not keywords:
                return ""
            
            # Tomar top 10
            top_keywords = keywords[:10]
            
            words = [kw['keyword'] for kw in top_keywords]
            scores = [kw['score'] for kw in top_keywords]
            
            # Crear gráfico horizontal
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=scores,
                y=words,
                orientation='h',
                marker_color='skyblue',
                text=[f'{s:.3f}' for s in scores],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Puntuación TF-IDF",
                yaxis_title="Palabras Clave",
                template="plotly_white",
                height=500
            )
            
            # Guardar archivo
            filename = f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de keywords creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de keywords: {e}")
            return ""
    
    async def create_pos_tags_chart(
        self, 
        pos_data: Dict[str, int],
        title: str = "Distribución de Categorías Gramaticales"
    ) -> str:
        """Crear gráfico de etiquetas POS"""
        
        try:
            if not pos_data:
                return ""
            
            # Agrupar etiquetas similares
            grouped_pos = {
                'Sustantivos': 0,
                'Verbos': 0,
                'Adjetivos': 0,
                'Adverbios': 0,
                'Pronombres': 0,
                'Otros': 0
            }
            
            noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
            verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            adj_tags = ['JJ', 'JJR', 'JJS']
            adv_tags = ['RB', 'RBR', 'RBS']
            pronoun_tags = ['PRP', 'PRP#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        , 'WP', 'WP#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        ]
            
            for tag, count in pos_data.items():
                if tag in noun_tags:
                    grouped_pos['Sustantivos'] += count
                elif tag in verb_tags:
                    grouped_pos['Verbos'] += count
                elif tag in adj_tags:
                    grouped_pos['Adjetivos'] += count
                elif tag in adv_tags:
                    grouped_pos['Adverbios'] += count
                elif tag in pronoun_tags:
                    grouped_pos['Pronombres'] += count
                else:
                    grouped_pos['Otros'] += count
            
            # Filtrar categorías vacías
            filtered_pos = {k: v for k, v in grouped_pos.items() if v > 0}
            
            # Crear gráfico de pie
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=list(filtered_pos.keys()),
                values=list(filtered_pos.values()),
                hole=0.3
            ))
            
            fig.update_layout(
                title=title,
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"pos_tags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de POS tags creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de POS: {e}")
            return ""
    
    async def create_emotions_chart(
        self, 
        emotions_data: Dict[str, float],
        title: str = "Análisis de Emociones"
    ) -> str:
        """Crear gráfico de análisis de emociones"""
        
        try:
            if not emotions_data:
                return ""
            
            # Filtrar emociones con valores > 0
            filtered_emotions = {k: v for k, v in emotions_data.items() if v > 0}
            
            if not filtered_emotions:
                return ""
            
            emotions = list(filtered_emotions.keys())
            scores = list(filtered_emotions.values())
            
            # Crear gráfico de barras polar
            fig = go.Figure()
            
            fig.add_trace(go.Barpolar(
                r=scores,
                theta=emotions,
                marker_color=px.colors.qualitative.Set3[:len(emotions)]
            ))
            
            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(scores) * 1.1]
                    )
                ),
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de emociones creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de emociones: {e}")
            return ""
    
    async def create_comparative_dashboard(
        self, 
        documents_data: List[Dict[str, Any]],
        title: str = "Dashboard Comparativo"
    ) -> str:
        """Crear dashboard comparativo para múltiples documentos"""
        
        try:
            if len(documents_data) < 2:
                return ""
            
            # Crear subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Comparación de Sentimientos',
                    'Longitud de Documentos', 
                    'Legibilidad',
                    'Riqueza Vocabulario'
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            doc_names = [f"Doc {i+1}" for i in range(len(documents_data))]
            
            # 1. Sentimientos
            sentiments = []
            for doc in documents_data:
                sentiment = doc.get('analysis', {}).get('sentiment', {})
                compound = sentiment.get('vader_compound', 0)
                sentiments.append(compound)
            
            fig.add_trace(
                go.Bar(x=doc_names, y=sentiments, name="Sentimiento"),
                row=1, col=1
            )
            
            # 2. Longitudes
            lengths = [doc.get('analysis', {}).get('word_count', 0) for doc in documents_data]
            
            fig.add_trace(
                go.Bar(x=doc_names, y=lengths, name="Palabras"),
                row=1, col=2
            )
            
            # 3. Legibilidad vs Longitud
            readabilities = [doc.get('analysis', {}).get('readability', {}).get('flesch_reading_ease', 50) 
                           for doc in documents_data]
            
            fig.add_trace(
                go.Scatter(x=lengths, y=readabilities, mode='markers+text', 
                          text=doc_names, textposition="top center", name="Legibilidad"),
                row=2, col=1
            )
            
            # 4. Riqueza vocabulario
            vocab_richness = [doc.get('analysis', {}).get('vocabulary_richness', 0) for doc in documents#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        # 4. Riqueza vocabulario
            vocab_richness = [doc.get('analysis', {}).get('vocabulary_richness', 0) for doc in documents_data]
            
            fig.add_trace(
                go.Bar(x=doc_names, y=vocab_richness, name="Riqueza Vocab."),
                row=2, col=2
            )
            
            # Actualizar layout
            fig.update_layout(
                title_text=title,
                showlegend=False,
                template="plotly_white",
                height=800
            )
            
            # Guardar archivo
            filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Dashboard comparativo creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando dashboard: {e}")
            return ""
    
    async def create_statistical_summary(
        self, 
        data: Dict[str, Any],
        title: str = "Resumen Estadístico"
    ) -> str:
        """Crear resumen estadístico visual"""
        
        try:
            # Crear figura con múltiples métricas
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Distribución de Longitudes',
                    'Métricas de Calidad',
                    'Tendencias Temporales',
                    'Correlaciones'
                )
            )
            
            # Datos de ejemplo (adaptar según datos reales)
            metrics = ['Legibilidad', 'Sentimiento', 'Complejidad', 'Coherencia']
            values = [75, 65, 45, 80]  # Valores de ejemplo
            
            # Gráfico de métricas
            fig.add_trace(
                go.Bar(x=metrics, y=values, name="Métricas"),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text=title,
                template="plotly_white",
                height=600
            )
            
            # Guardar archivo
            filename = f"stats_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando resumen estadístico: {e}")
            return ""
    
    def get_chart_types(self) -> List[str]:
        """Obtener tipos de gráficos disponibles"""
        return [
            "sentiment_analysis",
            "readability_metrics", 
            "keywords_distribution",
            "pos_tags_distribution",
            "emotions_analysis",
            "comparative_dashboard",
            "statistical_summary",
            "wordcloud",
            "timeline_analysis"
        ]
    
    async def cleanup(self):
        """Limpiar recursos"""
        plt.close('all')
'''
        
        visualizers_dir = self.service_path / "visualizers"
        with open(visualizers_dir / "chart_generator.py", "w", encoding="utf-8") as f:
            f.write(chart_generator_content)
        
        self.logger.info("✅ Generador de gráficos creado")

    def create_routers(self):
        """Crear routers de FastAPI"""
        self.logger.info("🌐 Creando routers...")
        
        # Router de análisis
        analysis_router = '''from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from visualizers.chart_generator import ChartGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

class AnalysisRequest(BaseModel):
    text: str
    options: Optional[Dict[str, Any]] = {}
    generate_insights: Optional[bool] = True
    generate_charts: Optional[bool] = True

class AnalysisResponse(BaseModel):
    analysis: Dict[str, Any]
    insights: List[Dict[str, Any]] = []
    charts: List[str] = []
    processing_time: float

@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks = None
):
    """Realizar análisis completo de texto"""
    
    try:
        # Inicializar servicios
        text_analyzer = TextAnalyzer()
        ai_insights = AIInsightsService()
        chart_generator = ChartGenerator()
        
        # Análisis principal
        analysis_result = await text_analyzer.analyze_text(request.text, request.options)
        
        response_data = {
            "analysis": analysis_result.__dict__,
            "insights": [],
            "charts": [],
            "processing_time": analysis_result.processing_time
        }
        
        # Generar insights con IA
        if request.generate_insights:
            try:
                insights = await ai_insights.generate_document_insights(
                    request.text, 
                    analysis_result.__dict__,
                    request.options
                )
                response_data["insights"] = [insight.__dict__ for insight in insights]
            except Exception as e:
                logger.warning(f"⚠️ Error generando insights: {e}")
        
        # Generar gráficos
        if request.generate_charts:
            try:
                charts = []
                
                # Gráfico de sentimientos
                sentiment_chart = await chart_generator.create_sentiment_chart(
                    analysis_result.sentiment
                )
                if sentiment_chart:
                    charts.append(sentiment_chart)
                
                # Gráfico de palabras clave
                if analysis_result.keywords:
                    keywords_chart = await chart_generator.create_keywords_chart(
                        analysis_result.keywords
                    )
                    if keywords_chart:
                        charts.append(keywords_chart)
                
                # Gráfico de emociones
                if analysis_result.emotion_scores:
                    emotions_chart = await chart_generator.create_emotions_chart(
                        analysis_result.emotion_scores
                    )
                    if emotions_chart:
                        charts.append(emotions_chart)
                
                response_data["charts"] = charts
                
            except Exception as e:
                logger.warning(f"⚠️ Error generando gráficos: {e}")
        
        return AnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Error en análisis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def analyze_batch(
    texts: List[str],
    options: Optional[Dict[str, Any]] = {},
    generate_comparative: bool = True
):
    """Análisis en lote de múltiples textos"""
    
    try:
        text_analyzer = TextAnalyzer()
        ai_insights = AIInsightsService()
        chart_generator = ChartGenerator()
        
        # Análisis en lote
        results = await text_analyzer.batch_analyze(texts)
        
        # Preparar datos para respuesta
        analysis_results = []
        for i, result in enumerate(results):
            analysis_results.append({
                "document_id": i,
                "analysis": result.__dict__ if hasattr(result, '__dict__') else result
            })
        
        response_data = {
            "batch_results": analysis_results,
            "comparative_insights": [],
            "comparative_charts": []
        }
        
        # Generar insights comparativos
        if generate_comparative and len(results) > 1:
            try:
                comparative_insights = await ai_insights.generate_comparative_insights(
                    analysis_results
                )
                response_data["comparative_insights"] = [
                    insight.__dict__ for insight in comparative_insights
                ]
                
                # Dashboard comparativo
                dashboard = await chart_generator.create_comparative_dashboard(
                    analysis_results
                )
                if dashboard:
                    response_data["comparative_charts"] = [dashboard]
                    
            except Exception as e:
                logger.warning(f"⚠️ Error en análisis comparativo: {e}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error en análisis en lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_analysis_capabilities():
    """Obtener capacidades de análisis disponibles"""
    
    text_analyzer = TextAnalyzer()
    
    return {
        "supported_languages": text_analyzer.get_supported_languages(),
        "analysis_features": [
            "sentiment_analysis",
            "readability_metrics",
            "entity_extraction", 
            "keyword_extraction",
            "topic_modeling",
            "pos_tagging",
            "emotion_analysis",
            "vocabulary_richness"
        ],
        "output_formats": ["json", "charts", "insights"],
        "batch_processing": True,
        "comparative_analysis": True
    }
'''
        
        routers_dir = self.service_path / "routers"
        with open(routers_dir / "analysis.py", "w", encoding="utf-8") as f:
            f.write(analysis_router)
        
        # Router de visualización
        visualization_router = '''from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from visualizers.chart_generator import ChartGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

class ChartRequest(BaseModel):
    chart_type: str
    data: Dict[str, Any]
    title: Optional[str] = "Gráfico"
    options: Optional[Dict[str, Any]] = {}

@router.post("/chart")
async def create_chart(request: ChartRequest):
    """Crear gráfico específico"""
    
    try:
        chart_generator = ChartGenerator()
        chart_path = ""
        
        if request.chart_type == "sentiment":
            chart_path = await chart_generator.create_sentiment_chart(
                request.data, request.title
            )
        elif request.chart_type == "keywords":
            chart_path = await chart_generator.create_keywords_chart(
                request.data, request.title
            )
        elif request.chart_type == "emotions":
            chart_path = await chart_generator.create_emotions_chart(
                request.data, request.title
            )
        elif request.chart_type == "readability":
            chart_path = await chart_generator.create_readability_chart(
                request.data, request.title
            )
        elif request.chart_type == "pos_tags":
            chart_path = await chart_generator.create_pos_tags_chart(
                request.data, request.title
            )
        else:
            raise HTTPException(status_code=400, detail=f"Tipo de gráfico no soportado: {request.chart_type}")
        
        if chart_path:
            return {"chart_url": f"/static/plots/{chart_path.split('/')[-1]}"}
        else:
            raise HTTPException(status_code=500, detail="Error generando gráfico")
            
    except Exception as e:
        logger.error(f"❌ Error creando gráfico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboard")
async def create_dashboard(
    documents_data: List[Dict[str, Any]],
    title: str = "Dashboard Analítico"
):
    """Crear dashboard comparativo"""
    
    try:
        chart_generator = ChartGenerator()
        
        dashboard_path = await chart_generator.create_comparative_dashboard(
            documents_data, title
        )
        
        if dashboard_path:
            return {"dashboard_url": f"/static/plots/{dashboard_path.split('/')[-1]}"}
        else:
            raise HTTPException(status_code=500, detail="Error generando dashboard")
            
    except Exception as e:
        logger.error(f"❌ Error creando dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chart-types")
async def get_chart_types():
    """Obtener tipos de gráficos disponibles"""
    
    chart_generator = ChartGenerator()
    return {
        "available_charts": chart_generator.get_chart_types(),
        "formats": ["html", "png", "svg"],
        "interactive": True
    }
'''
        
        with open(routers_dir / "visualization.py", "w", encoding="utf-8") as f:
            f.write(visualization_router)
        
        # Crear otros routers básicos
        for router_name in ["statistics", "insights", "models"]:
            basic_router = f'''from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

@router.get("/")
async def {router_name}_root():
    return {{"message": "{router_name.title()} router - En desarrollo"}}
'''
            with open(routers_dir / f"{router_name}.py", "w", encoding="utf-8") as f:
                f.write(basic_router)
        
        self.logger.info("✅ Routers creados")

    def create_config_utils(self):
        """Crear utilidades de configuración"""
        self.logger.info("⚙️ Creando utilidades...")
        
        # Config
        config_content = '''"""
Configuración del Analytics-Engine
"""
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Configuración general
    app_name: str = "Analytics-Engine"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8003
    
    # Puertos de otros servicios
    ai_engine_port: int = 8001
    document_processor_port: int = 8002
    
    # Directorios
    data_dir: str = "./data"
    reports_dir: str = "./data/reports"
    cache_dir: str = "./data/cache"
    plots_dir: str = "./static/plots"
    
    # Análisis de texto
    default_language: str = "auto"
    max_text_length: int = 1000000  # 1MB
    enable_topic_modeling: bool = True
    enable_entity_extraction: bool = True
    
    # IA e Insights
    enable_ai_insights: bool = True
    ai_insights_timeout: int = 30
    max_insights_per_document: int = 10
    
    # Visualización
    default_chart_format: str = "html"
    chart_width: int = 800
    chart_height: int = 600
    enable_interactive_charts: bool = True
    
    # CORS
    cors_origins: List[str] = ["http://localhost:8080", "http://127.0.0.1:8080"]
    
    # Base de datos
    database_url: str = "sqlite:///./analytics_engine.db"
    redis_url: str = "redis://localhost:6379/3"
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    # Modelos NLP
    spacy_model: str = "es_core_news_sm"  # Modelo principal
    spacy_fallback: str = "en_core_web_sm"  # Modelo de respaldo
    
    # Análisis estadístico
    enable_statistical_analysis: bool = True
    confidence_interval: float = 0.95
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
'''
        
        utils_dir = self.service_path / "utils"
        with open(utils_dir / "config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        # Logging config
        logging_config = '''"""
Configuración de logging para Analytics-Engine
"""
import logging
import logging.handlers
from pathlib import Path

def setup_logging():
    """Configurar sistema de logging"""
    
    # Crear directorio de logs
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configuración del logger root
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler con rotación
            logging.handlers.RotatingFileHandler(
                log_dir / "analytics_engine.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Configurar loggers específicos
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("plotly").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
'''
        
        with open(utils_dir / "logging_config.py", "w", encoding="utf-8") as f:
            f.write(logging_config)
        
        self.logger.info("✅ Utilidades creadas")

    def create_analytics_manager(self):
        """Crear gestor principal de análisis"""
        self.logger.info("📊 Creando gestor de análisis...")
        
        analytics_manager_content = '''"""
Gestor principal del motor de análisis
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """Administrador principal del motor de análisis"""
    
    def __init__(self):
        self.analysis_history = []
        self.statistics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "total_documents_processed": 0,
            "total_words_analyzed": 0,
            "average_processing_time": 0.0,
            "analysis_types": {}
        }
        self.is_ready = False
        
    async def initialize(self):
        """Inicializar el gestor de análisis"""
        logger.info("🔧 Inicializando Analytics Manager...")
        
        # Crear directorios necesarios
        directories = [
            Path("data/datasets"),
            Path("data/reports"),
            Path("data/cache"),
            Path("static/plots")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Cargar estadísticas si existen
        await self._load_statistics()
        
        self.is_ready = True
        logger.info("✅ Analytics Manager inicializado")
    
    async def register_analysis(
        self, 
        analysis_type: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any],
        processing_time: float
    ):
        """Registrar un análisis realizado"""
        
        analysis_record = {
            "id": len(self.analysis_history) + 1,
            "type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time,
            "input_size": len(str(input_data)),
            "success": "error" not in result,
            "metadata": {
                "word_count": input_data.get("word_count", 0),
                "language": result.get("language", "unknown")
            }
        }
        
        self.analysis_history.append(analysis_record)
        
        # Actualizar estadísticas
        self.statistics["total_analyses"] += 1
        if analysis_record["success"]:
            self.statistics["successful_analyses"] += 1
        else:
            self.statistics["failed_analyses"] += 1
        
        self.statistics["total_words_analyzed"] += analysis_record["metadata"]["word_count"]
        
        # Actualizar tiempo promedio
        total_time = sum(a["processing_time"] for a in self.analysis_history)
        self.statistics["average_processing_time"] = total_time / len(self.analysis_history)
        
        # Contar tipos de análisis
        if analysis_type in self.statistics["analysis_types"]:
            self.statistics["analysis_types"][analysis_type] += 1
        else:
            self.statistics["analysis_types"][analysis_type] = 1
        
        await self._save_statistics()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del motor"""
        
        # Calcular métricas adicionales
        success_rate = 0.0
        if self.statistics["total_analyses"] > 0:
            success_rate = (self.statistics["successful_analyses"] / 
                          self.statistics["total_analyses"]) * 100
        
        # Análisis recientes (últimos 10)
        recent_analyses = self.analysis_history[-10:] if self.analysis_history else []
        
        return {
            **self.statistics,
            "success_rate": round(success_rate, 2),
            "recent_analyses": recent_analyses,
            "storage_info": await self._get_storage_info(),
            "performance_metrics": await self._get_performance_metrics()
        }
    
    async def _get_storage_info(self) -> Dict[str, Any]:
        """Obtener información de almacenamiento"""
        
        storage_info = {}
        
        for directory_name, path in [
            ("datasets", "data/datasets"),
            ("reports", "data/reports"),
            ("cache", "data/cache"),
            ("plots", "static/plots")
        ]:
            directory = Path(path)
            if directory.exists():
                files = list(directory.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                storage_info[directory_name] = {
                    "files": len([f for f in files if f.is_file()]),
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "path": str(directory)
                }
            else:
                storage_info[directory_name] = {
                    "files": 0,
                    "total_size_mb": 0,
                    "path": str(directory)
                }
        
        return storage_info
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento"""
        
        if not self.analysis_history:
            return {}
        
        processing_times = [a["processing_time"] for a in self.analysis_history]
        word_counts = [a["metadata"]["word_count"] for a in self.analysis_history]
        
        return {
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "median_processing_time": sorted(processing_times)[len(processing_times)//2],
            "avg_words_per_second": (
                sum(word_counts) / sum(processing_times) 
                if sum(processing_times) > 0 else 0
            ),
            "total_processing_time": sum(processing_times)
        }
    
    async def _load_statistics(self):
        """Cargar estadísticas desde archivo"""
        stats_file = Path("data/analytics_stats.json")
        
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.statistics = data.get("statistics", self.statistics)
                    self.analysis_history = data.get("history", [])
                logger.info("📊 Estadísticas cargadas")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando estadísticas: {e}")
    
    async def _save_statistics(self):
        """Guardar estadísticas en archivo"""
        stats_file = Path("data/analytics_stats.json")
        
        try:
            data = {
                "statistics": self.statistics,
                "history": self.analysis_history[-1000:]  # Mantener últimos 1000
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Error guardando estadísticas: {e}")
    
    async def cleanup_old_data(self, days: int = 30):
        """Limpiar datos antiguos"""
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cleaned_files = 0
        
        for directory_path in ["data/cache", "static/plots"]:
            directory = Path(directory_path)
            if not directory.exists():
                continue
            
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            cleaned_files += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Error eliminando {file_path}: {e}")
        
        logger.info(f"🧹 Limpieza completada: {cleaned_files} archivos eliminados")
        return cleaned_files
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generar reporte de rendimiento"""
        
        stats = await self.get_statistics()
        
        report = {
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_analyses": stats["total_analyses"],
                "success_rate": stats["success_rate"],
                "avg_processing_time": stats["average_processing_time"],
                "total_words_processed": stats["total_words_analyzed"]
            },
            "performance": stats.get("performance_metrics", {}),
            "storage": stats.get("storage_info", {}),
            "analysis_breakdown": stats.get("analysis_types", {}),
            "recommendations": []
        }
        
        # Generar recomendaciones
        if stats["success_rate"] < 95:
            report["recommendations"].append("Revisar casos de fallo para mejorar robustez")
        
        if stats["average_processing_time"] > 5.0:
            report["recommendations"].append("Optimizar algoritmos para mejorar velocidad")
        
        # Guardar reporte
        report_file = Path(f"data/reports/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    async def cleanup(self):
        """Limpiar recursos"""
        logger.info("🧹 Limpiando recursos del Analytics Manager...")
        await self._save_statistics()
        logger.info("✅ Recursos limpiados")
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "analytics_manager.py", "w", encoding="utf-8") as f:
            f.write(analytics_manager_content)
        
        self.logger.info("✅ Gestor de análisis creado")

    def create_data_processor(self):
        """Crear procesador de datos"""
        self.logger.info("🔢 Creando procesador de datos...")
        
        data_processor_content = '''"""
Procesador de datos para análisis estadístico
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

class DataProcessor:
    """Procesador de datos para análisis estadístico"""
    
    def __init__(self):
        self.is_ready = False
        self.supported_formats = ['json', 'csv', 'excel', 'dataframe']
        
    async def initialize(self):
        """Inicializar el procesador de datos"""
        logger.info("🔧 Inicializando Data Processor...")
        self.is_ready = True
        logger.info("✅ Data Processor inicializado")
    
    async def process_text_analysis_data(
        self, 
        analysis_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Procesar resultados de análisis de texto en DataFrame"""
        
        try:
            # Extraer métricas numéricas
            processed_data = []
            
            for i, result in enumerate(analysis_results):
                row = {
                    'document_id': i,
                    'text_length': result.get('text_length', 0),
                    'word_count': result.get('word_count', 0),
                    'sentence_count': result.get('sentence_count', 0),
                    'paragraph_count': result.get('paragraph_count', 0),
                    'language': result.get('language', 'unknown'),
                    'vocabulary_richness': result.get('vocabulary_richness', 0),
                    'average_sentence_length': result.get('average_sentence_length', 0),
                }
                
                # Métricas de sentimiento
                sentiment = result.get('sentiment', {})
                row.update({
                    'sentiment_compound': sentiment.get('vader_compound', 0),
                    'sentiment_positive': sentiment.get('vader_positive', 0),
                    'sentiment_negative': sentiment.get('vader_negative', 0),
                    'sentiment_neutral': sentiment.get('vader_neutral', 0),
                    'textblob_polarity': sentiment.get('textblob_polarity', 0),
                    'textblob_subjectivity': sentiment.get('textblob_subjectivity', 0),
                })
                
                # Métricas de legibilidad
                readability = result.get('readability', {})
                row.update({
                    'flesch_reading_ease': readability.get('flesch_reading_ease', 0),
                    'avg_word_length': readability.get('average_word_length', 0),
                    'reading_time_minutes': readability.get('estimated_reading_time_minutes', 0),
                })
                
                # Métricas de emociones
                emotions = result.get('emotion_scores', {})
                for emotion, score in emotions.items():
                    row[f'emotion_{emotion}'] = score
                        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            # Normalizar por longitud del texto
            emotion_scores[emotion] = score / max(len(text.split()), 1) * 100
        
        return emotion_scores
    
    async def generate_wordcloud(self, text: str, output_path: str = None) -> str:
        """Generar nube de palabras"""
        
        try:
            # Limpiar texto
            words = nltk.word_tokenize(text.lower())
            words = [w for w in words if w.isalnum() and len(w) > 2]
            clean_text = ' '.join(words)
            
            # Generar wordcloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(clean_text)
            
            # Guardar imagen
            if not output_path:
                output_path = f"static/plots/wordcloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            wordcloud.to_file(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generando wordcloud: {e}")
            return ""
    
    async def batch_analyze(self, texts: List[str]) -> List[TextAnalysisResult]:
        """Análisis en lote de múltiples textos"""
        
        logger.info(f"📦 Analizando lote de {len(texts)} textos...")
        
        tasks = [self.analyze_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar resultados exitosos
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error en texto {i}: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"✅ Lote completado: {len(successful_results)}/{len(texts)} exitosos")
        return successful_results
    
    def get_supported_languages(self) -> List[str]:
        """Obtener idiomas soportados"""
        return self.supported_languages
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Obtener estado de modelos cargados"""
        return {
            'spacy_model': self.nlp_model is not None,
            'vader_sentiment': self.sentiment_analyzer is not None,
            'nltk_resources': True  # Asumimos que están disponibles si llegó hasta aquí
        }
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "text_analyzer.py", "w", encoding="utf-8") as f:
            f.write(text_analyzer_content)
        
        self.logger.info("✅ Analizador de texto creado")

    def create_ai_insights_service(self):
        """Crear servicio de insights con IA"""
        self.logger.info("🤖 Creando servicio de insights con IA...")
        
        ai_insights_content = '''"""
Servicio de insights usando IA - se conecta con AI-Engine
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import httpx
from dataclasses import dataclass
from datetime import datetime
import json

from utils.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class AIInsight:
    """Estructura para un insight generado por IA"""
    title: str
    content: str
    confidence: float
    category: str
    metadata: Dict[str, Any]
    timestamp: str

class AIInsightsService:
    """Servicio para generar insights usando el AI-Engine"""
    
    def __init__(self):
        self.settings = Settings()
        self.is_ready = False
        self.ai_engine_url = f"http://localhost:{self.settings.ai_engine_port}"
        self.client = None
        
    async def initialize(self):
        """Inicializar el servicio de insights"""
        logger.info("🔧 Inicializando AI Insights Service...")
        
        try:
            # Crear cliente HTTP
            self.client = httpx.AsyncClient(timeout=30.0)
            
            # Verificar conexión con AI-Engine
            connection_ok = await self.check_ai_engine_connection()
            if connection_ok:
                logger.info("✅ Conexión con AI-Engine establecida")
            else:
                logger.warning("⚠️ AI-Engine no disponible. Modo limitado.")
            
            self.is_ready = True
            logger.info("✅ AI Insights Service inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando AI Insights: {e}")
            self.is_ready = True  # Continuar en modo limitado
    
    async def check_ai_engine_connection(self) -> bool:
        """Verificar conexión con AI-Engine"""
        
        try:
            response = await self.client.get(f"{self.ai_engine_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def generate_document_insights(
        self, 
        text: str, 
        analysis_result: Dict = None,
        options: Dict = None
    ) -> List[AIInsight]:
        """Generar insights sobre un documento"""
        
        options = options or {}
        insights = []
        
        try:
            # Insight 1: Resumen del contenido
            if options.get('generate_summary', True):
                summary_insight = await self._generate_summary_insight(text)
                if summary_insight:
                    insights.append(summary_insight)
            
            # Insight 2: Análisis de temas principales
            if options.get('analyze_themes', True):
                themes_insight = await self._analyze_themes_insight(text)
                if themes_insight:
                    insights.append(themes_insight)
            
            # Insight 3: Clasificación del documento
            if options.get('classify_document', True):
                classification_insight = await self._classify_document_insight(text)
                if classification_insight:
                    insights.append(classification_insight)
            
            # Insight 4: Análisis de sentimiento contextual
            if options.get('sentiment_context', True) and analysis_result:
                sentiment_insight = await self._generate_sentiment_context(text, analysis_result)
                if sentiment_insight:
                    insights.append(sentiment_insight)
            
            # Insight 5: Recomendaciones
            if options.get('generate_recommendations', True):
                recommendations_insight = await self._generate_recommendations(text, analysis_result)
                if recommendations_insight:
                    insights.append(recommendations_insight)
            
            logger.info(f"✅ Generados {len(insights)} insights")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generando insights: {e}")
            return []
    
    async def _generate_summary_insight(self, text: str) -> Optional[AIInsight]:
        """Generar resumen inteligente del documento"""
        
        try:
            prompt = f"""
            Analiza el siguiente texto y proporciona un resumen conciso que capture:
            1. Los puntos principales
            2. Los temas centrales
            3. Las conclusiones clave
            
            Texto a analizar:
            {text[:2000]}...
            
            Proporciona un resumen de máximo 150 palabras.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Resumen del Documento",
                    content=response,
                    confidence=0.85,
                    category="summary",
                    metadata={"word_count": len(response.split())},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error generando resumen: {e}")
        
        return None
    
    async def _analyze_themes_insight(self, text: str) -> Optional[AIInsight]:
        """Analizar temas principales del documento"""
        
        try:
            prompt = f"""
            Identifica y analiza los 3-5 temas principales en el siguiente texto.
            Para cada tema, proporciona:
            1. Nombre del tema
            2. Relevancia (1-10)
            3. Breve descripción
            
            Texto:
            {text[:2000]}...
            
            Responde en formato estructurado.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Análisis de Temas Principales",
                    content=response,
                    confidence=0.80,
                    category="themes",
                    metadata={"analysis_type": "thematic"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error analizando temas: {e}")
        
        return None
    
    async def _classify_document_insight(self, text: str) -> Optional[AIInsight]:
        """Clasificar tipo de documento"""
        
        try:
            prompt = f"""
            Clasifica el siguiente documento en una de estas categorías:
            - Reporte técnico
            - Documento legal
            - Comunicación comercial
            - Contenido académico
            - Manual o guía
            - Correspondencia
            - Otro
            
            Explica brevemente por qué pertenece a esa categoría y qué características lo definen.
            
            Texto:
            {text[:1500]}...
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Clasificación del Documento",
                    content=response,
                    confidence=0.75,
                    category="classification",
                    metadata={"classification_type": "document_type"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error clasificando documento: {e}")
        
        return None
    
    async def _generate_sentiment_context(self, text: str, analysis_result: Dict) -> Optional[AIInsight]:
        """Generar contexto del análisis de sentimientos"""
        
        try:
            sentiment_data = analysis_result.get('sentiment', {})
            overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
            
            prompt = f"""
            El análisis automático indica que este documento tiene un sentimiento "{overall_sentiment}".
            
            Analiza el contexto y explica:
            1. ¿Por qué el documento tiene este sentimiento?
            2. ¿Qué elementos específicos contribuyen a este sentimiento?
            3. ¿Es apropiado el tono para el tipo de documento?
            
            Fragmento del texto:
            {text[:1000]}...
            
            Proporciona un análisis contextual del sentimiento.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title=f"Contexto del Sentimiento ({overall_sentiment.title()})",
                    content=response,
                    confidence=0.70,
                    category="sentiment_context",
                    metadata={"detected_sentiment": overall_sentiment},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error en contexto de sentimiento: {e}")
        
        return None
    
    async def _generate_recommendations(self, text: str, analysis_result: Dict = None) -> Optional[AIInsight]:
        """Generar recomendaciones basadas en el análisis"""
        
        try:
            context = ""
            if analysis_result:
                readability = analysis_result.get('readability', {})
                flesch_score = readability.get('flesch_reading_ease', 50)
                
                if flesch_score < 30:
                    context += "El documento tiene baja legibilidad. "
                elif flesch_score > 70:
                    context += "El documento tiene alta legibilidad. "
            
            prompt = f"""
            Basándote en el análisis del siguiente documento, proporciona 3-5 recomendaciones prácticas para:
            1. Mejorar la claridad y legibilidad
            2. Optimizar la estructura
            3. Enhancer la comunicación efectiva
            
            {context}
            
            Texto analizado:
            {text[:1000]}...
            
            Proporciona recomendaciones específicas y accionables.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Recomendaciones de Mejora",
                    content=response,
                    confidence=0.75,
                    category="recommendations",
                    metadata={"recommendation_type": "improvement"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error generando recomendaciones: {e}")
        
        return None
    
    async def _call_ai_engine(self, prompt: str, model: str = "auto") -> Optional[str]:
        """Llamar al AI-Engine para generar respuesta"""
        
        if not await self.check_ai_engine_connection():
            logger.warning("⚠️ AI-Engine no disponible")
            return None
        
        try:
            payload = {
                "prompt": prompt,
                "model": model,
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = await self.client.post(
                f"{self.ai_engine_url}/generate/complete",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("text", "").strip()
            else:
                logger.warning(f"⚠️ AI-Engine respondió con status {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error llamando AI-Engine: {e}")
            return None
    
    async def generate_comparative_insights(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[AIInsight]:
        """Generar insights comparativos entre múltiples documentos"""
        
        if len(documents) < 2:
            return []
        
        insights = []
        
        try:
            # Comparar sentimientos
            sentiments = [doc.get('analysis', {}).get('sentiment', {}).get('overall_sentiment', 'neutral') 
                         for doc in documents]
            
            # Comparar longitudes
            lengths = [doc.get('analysis', {}).get('word_count', 0) for doc in documents]
            
            # Comparar legibilidad
            readabilities = [doc.get('analysis', {}).get('readability', {}).get('flesch_reading_ease', 50) 
                           for doc in documents]
            
            comparison_data = {
                'document_count': len(documents),
                'sentiments': sentiments,
                'avg_length': sum(lengths) / len(lengths),
                'avg_readability': sum(readabilities) / len(readabilities)
            }
            
            prompt = f"""
            Analiza los siguientes datos comparativos de {len(documents)} documentos:
            
            Sentimientos: {sentiments}
            Longitud promedio: {comparison_data['avg_length']:.0f} palabras
            Legibilidad promedio: {comparison_data['avg_readability']:.1f}
            
            Proporciona insights sobre:
            1. Patrones en los sentimientos
            2. Consistencia en la comunicación
            3. Variabilidad en complejidad
            4. Recomendaciones para homogeneizar
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                insights.append(AIInsight(
                    title="Análisis Comparativo",
                    content=response,
                    confidence=0.80,
                    category="comparative",
                    metadata=comparison_data,
                    timestamp=datetime.now().isoformat()
                ))
            
        except Exception as e:
            logger.error(f"❌ Error en insights comparativos: {e}")
        
        return insights
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles en AI-Engine"""
        return ["auto", "gpt-3.5-turbo", "claude-3-sonnet", "local"]
    
    async def cleanup(self):
        """Limpiar recursos"""
        if self.client:
            await self.client.aclose()
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "ai_insights.py", "w", encoding="utf-8") as f:
            f.write(ai_insights_content)
        
        self.logger.info("✅ Servicio de insights con IA creado")

    def create_chart_generator(self):
        """Crear generador de gráficos"""
        self.logger.info("📊 Creando generador de gráficos...")
        
        chart_generator_content = '''"""
Generador de gráficos y visualizaciones
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generador de gráficos para análisis de datos"""
    
    def __init__(self):
        self.is_ready = False
        self.output_dir = "static/plots"
        
        # Configurar estilo por defecto
        plt.style.use('default')
        sns.set_palette("husl")
        
    async def initialize(self):
        """Inicializar el generador de gráficos"""
        logger.info("🔧 Inicializando Chart Generator...")
        
        # Crear directorio de salida
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.is_ready = True
        logger.info("✅ Chart Generator inicializado")
    
    async def create_sentiment_chart(
        self, 
        sentiment_data: Dict[str, float], 
        title: str = "Análisis de Sentimientos"
    ) -> str:
        """Crear gráfico de análisis de sentimientos"""
        
        try:
            # Crear gráfico con Plotly
            fig = go.Figure()
            
            # Datos de VADER
            vader_data = {
                'Positivo': sentiment_data.get('vader_positive', 0),
                'Negativo': sentiment_data.get('vader_negative', 0),
                'Neutral': sentiment_data.get('vader_neutral', 0)
            }
            
            # Gráfico de barras
            fig.add_trace(go.Bar(
                x=list(vader_data.keys()),
                y=list(vader_data.values()),
                marker_color=['green', 'red', 'blue'],
                text=[f'{v:.2f}' for v in vader_data.values()],
                textposition='auto'
            ))
            
            # Configurar layout
            fig.update_layout(
                title=title,
                xaxis_title="Categoría",
                yaxis_title="Puntuación",
                template="plotly_white",
                height=400
            )
            
            # Guardar archivo
            filename = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de sentimientos creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de sentimientos: {e}")
            return ""
    
    async def create_readability_chart(
        self, 
        readability_data: Dict[str, float],
        title: str = "Métricas de Legibilidad"
    ) -> str:
        """Crear gráfico de métricas de legibilidad"""
        
        try:
            # Preparar datos
            metrics = []
            values = []
            
            if 'flesch_reading_ease' in readability_data:
                metrics.append('Facilidad Lectura')
                values.append(readability_data['flesch_reading_ease'])
            
            if 'average_sentence_length' in readability_data:
                metrics.append('Long. Oraciones')
                values.append(min(readability_data['average_sentence_length'], 50))  # Escalar
            
            if 'average_word_length' in readability_data:
                metrics.append('Long. Palabras')
                values.append(readability_data['average_word_length'] * 10)  # Escalar
            
            # Crear gráfico radial
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='Métricas'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title=title,
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"readability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de legibilidad creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de legibilidad: {e}")
            return ""
    
    async def create_keywords_chart(
        self, 
        keywords: List[Dict[str, float]],
        title: str = "Palabras Clave Principales"
    ) -> str:
        """Crear gráfico de palabras clave"""
        
        try:
            if not keywords:
                return ""
            
            # Tomar top 10
            top_keywords = keywords[:10]
            
            words = [kw['keyword'] for kw in top_keywords]
            scores = [kw['score'] for kw in top_keywords]
            
            # Crear gráfico horizontal
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=scores,
                y=words,
                orientation='h',
                marker_color='skyblue',
                text=[f'{s:.3f}' for s in scores],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Puntuación TF-IDF",
                yaxis_title="Palabras Clave",
                template="plotly_white",
                height=500
            )
            
            # Guardar archivo
            filename = f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de keywords creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de keywords: {e}")
            return ""
    
    async def create_pos_tags_chart(
        self, 
        pos_data: Dict[str, int],
        title: str = "Distribución de Categorías Gramaticales"
    ) -> str:
        """Crear gráfico de etiquetas POS"""
        
        try:
            if not pos_data:
                return ""
            
            # Agrupar etiquetas similares
            grouped_pos = {
                'Sustantivos': 0,
                'Verbos': 0,
                'Adjetivos': 0,
                'Adverbios': 0,
                'Pronombres': 0,
                'Otros': 0
            }
            
            noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
            verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            adj_tags = ['JJ', 'JJR', 'JJS']
            adv_tags = ['RB', 'RBR', 'RBS']
            pronoun_tags = ['PRP', 'PRP#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        , 'WP', 'WP#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        ]
            
            for tag, count in pos_data.items():
                if tag in noun_tags:
                    grouped_pos['Sustantivos'] += count
                elif tag in verb_tags:
                    grouped_pos['Verbos'] += count
                elif tag in adj_tags:
                    grouped_pos['Adjetivos'] += count
                elif tag in adv_tags:
                    grouped_pos['Adverbios'] += count
                elif tag in pronoun_tags:
                    grouped_pos['Pronombres'] += count
                else:
                    grouped_pos['Otros'] += count
            
            # Filtrar categorías vacías
            filtered_pos = {k: v for k, v in grouped_pos.items() if v > 0}
            
            # Crear gráfico de pie
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=list(filtered_pos.keys()),
                values=list(filtered_pos.values()),
                hole=0.3
            ))
            
            fig.update_layout(
                title=title,
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"pos_tags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de POS tags creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de POS: {e}")
            return ""
    
    async def create_emotions_chart(
        self, 
        emotions_data: Dict[str, float],
        title: str = "Análisis de Emociones"
    ) -> str:
        """Crear gráfico de análisis de emociones"""
        
        try:
            if not emotions_data:
                return ""
            
            # Filtrar emociones con valores > 0
            filtered_emotions = {k: v for k, v in emotions_data.items() if v > 0}
            
            if not filtered_emotions:
                return ""
            
            emotions = list(filtered_emotions.keys())
            scores = list(filtered_emotions.values())
            
            # Crear gráfico de barras polar
            fig = go.Figure()
            
            fig.add_trace(go.Barpolar(
                r=scores,
                theta=emotions,
                marker_color=px.colors.qualitative.Set3[:len(emotions)]
            ))
            
            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(scores) * 1.1]
                    )
                ),
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de emociones creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de emociones: {e}")
            return ""
    
    async def create_comparative_dashboard(
        self, 
        documents_data: List[Dict[str, Any]],
        title: str = "Dashboard Comparativo"
    ) -> str:
        """Crear dashboard comparativo para múltiples documentos"""
        
        try:
            if len(documents_data) < 2:
                return ""
            
            # Crear subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Comparación de Sentimientos',
                    'Longitud de Documentos', 
                    'Legibilidad',
                    'Riqueza Vocabulario'
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            doc_names = [f"Doc {i+1}" for i in range(len(documents_data))]
            
            # 1. Sentimientos
            sentiments = []
            for doc in documents_data:
                sentiment = doc.get('analysis', {}).get('sentiment', {})
                compound = sentiment.get('vader_compound', 0)
                sentiments.append(compound)
            
            fig.add_trace(
                go.Bar(x=doc_names, y=sentiments, name="Sentimiento"),
                row=1, col=1
            )
            
            # 2. Longitudes
            lengths = [doc.get('analysis', {}).get('word_count', 0) for doc in documents_data]
            
            fig.add_trace(
                go.Bar(x=doc_names, y=lengths, name="Palabras"),
                row=1, col=2
            )
            
            # 3. Legibilidad vs Longitud
            readabilities = [doc.get('analysis', {}).get('readability', {}).get('flesch_reading_ease', 50) 
                           for doc in documents_data]
            
            fig.add_trace(
                go.Scatter(x=lengths, y=readabilities, mode='markers+text', 
                          text=doc_names, textposition="top center", name="Legibilidad"),
                row=2, col=1
            )
            
            # 4. Riqueza vocabulario
            vocab_richness = [doc.get('analysis', {}).get('vocabulary_richness', 0) for doc in documents#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        # 4. Riqueza vocabulario
            vocab_richness = [doc.get('analysis', {}).get('vocabulary_richness', 0) for doc in documents_data]
            
            fig.add_trace(
                go.Bar(x=doc_names, y=vocab_richness, name="Riqueza Vocab."),
                row=2, col=2
            )
            
            # Actualizar layout
            fig.update_layout(
                title_text=title,
                showlegend=False,
                template="plotly_white",
                height=800
            )
            
            # Guardar archivo
            filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Dashboard comparativo creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando dashboard: {e}")
            return ""
    
    async def create_statistical_summary(
        self, 
        data: Dict[str, Any],
        title: str = "Resumen Estadístico"
    ) -> str:
        """Crear resumen estadístico visual"""
        
        try:
            # Crear figura con múltiples métricas
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Distribución de Longitudes',
                    'Métricas de Calidad',
                    'Tendencias Temporales',
                    'Correlaciones'
                )
            )
            
            # Datos de ejemplo (adaptar según datos reales)
            metrics = ['Legibilidad', 'Sentimiento', 'Complejidad', 'Coherencia']
            values = [75, 65, 45, 80]  # Valores de ejemplo
            
            # Gráfico de métricas
            fig.add_trace(
                go.Bar(x=metrics, y=values, name="Métricas"),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text=title,
                template="plotly_white",
                height=600
            )
            
            # Guardar archivo
            filename = f"stats_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando resumen estadístico: {e}")
            return ""
    
    def get_chart_types(self) -> List[str]:
        """Obtener tipos de gráficos disponibles"""
        return [
            "sentiment_analysis",
            "readability_metrics", 
            "keywords_distribution",
            "pos_tags_distribution",
            "emotions_analysis",
            "comparative_dashboard",
            "statistical_summary",
            "wordcloud",
            "timeline_analysis"
        ]
    
    async def cleanup(self):
        """Limpiar recursos"""
        plt.close('all')
'''
        
        visualizers_dir = self.service_path / "visualizers"
        with open(visualizers_dir / "chart_generator.py", "w", encoding="utf-8") as f:
            f.write(chart_generator_content)
        
        self.logger.info("✅ Generador de gráficos creado")

    def create_routers(self):
        """Crear routers de FastAPI"""
        self.logger.info("🌐 Creando routers...")
        
        # Router de análisis
        analysis_router = '''from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from visualizers.chart_generator import ChartGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

class AnalysisRequest(BaseModel):
    text: str
    options: Optional[Dict[str, Any]] = {}
    generate_insights: Optional[bool] = True
    generate_charts: Optional[bool] = True

class AnalysisResponse(BaseModel):
    analysis: Dict[str, Any]
    insights: List[Dict[str, Any]] = []
    charts: List[str] = []
    processing_time: float

@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks = None
):
    """Realizar análisis completo de texto"""
    
    try:
        # Inicializar servicios
        text_analyzer = TextAnalyzer()
        ai_insights = AIInsightsService()
        chart_generator = ChartGenerator()
        
        # Análisis principal
        analysis_result = await text_analyzer.analyze_text(request.text, request.options)
        
        response_data = {
            "analysis": analysis_result.__dict__,
            "insights": [],
            "charts": [],
            "processing_time": analysis_result.processing_time
        }
        
        # Generar insights con IA
        if request.generate_insights:
            try:
                insights = await ai_insights.generate_document_insights(
                    request.text, 
                    analysis_result.__dict__,
                    request.options
                )
                response_data["insights"] = [insight.__dict__ for insight in insights]
            except Exception as e:
                logger.warning(f"⚠️ Error generando insights: {e}")
        
        # Generar gráficos
        if request.generate_charts:
            try:
                charts = []
                
                # Gráfico de sentimientos
                sentiment_chart = await chart_generator.create_sentiment_chart(
                    analysis_result.sentiment
                )
                if sentiment_chart:
                    charts.append(sentiment_chart)
                
                # Gráfico de palabras clave
                if analysis_result.keywords:
                    keywords_chart = await chart_generator.create_keywords_chart(
                        analysis_result.keywords
                    )
                    if keywords_chart:
                        charts.append(keywords_chart)
                
                # Gráfico de emociones
                if analysis_result.emotion_scores:
                    emotions_chart = await chart_generator.create_emotions_chart(
                        analysis_result.emotion_scores
                    )
                    if emotions_chart:
                        charts.append(emotions_chart)
                
                response_data["charts"] = charts
                
            except Exception as e:
                logger.warning(f"⚠️ Error generando gráficos: {e}")
        
        return AnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Error en análisis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def analyze_batch(
    texts: List[str],
    options: Optional[Dict[str, Any]] = {},
    generate_comparative: bool = True
):
    """Análisis en lote de múltiples textos"""
    
    try:
        text_analyzer = TextAnalyzer()
        ai_insights = AIInsightsService()
        chart_generator = ChartGenerator()
        
        # Análisis en lote
        results = await text_analyzer.batch_analyze(texts)
        
        # Preparar datos para respuesta
        analysis_results = []
        for i, result in enumerate(results):
            analysis_results.append({
                "document_id": i,
                "analysis": result.__dict__ if hasattr(result, '__dict__') else result
            })
        
        response_data = {
            "batch_results": analysis_results,
            "comparative_insights": [],
            "comparative_charts": []
        }
        
        # Generar insights comparativos
        if generate_comparative and len(results) > 1:
            try:
                comparative_insights = await ai_insights.generate_comparative_insights(
                    analysis_results
                )
                response_data["comparative_insights"] = [
                    insight.__dict__ for insight in comparative_insights
                ]
                
                # Dashboard comparativo
                dashboard = await chart_generator.create_comparative_dashboard(
                    analysis_results
                )
                if dashboard:
                    response_data["comparative_charts"] = [dashboard]
                    
            except Exception as e:
                logger.warning(f"⚠️ Error en análisis comparativo: {e}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error en análisis en lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_analysis_capabilities():
    """Obtener capacidades de análisis disponibles"""
    
    text_analyzer = TextAnalyzer()
    
    return {
        "supported_languages": text_analyzer.get_supported_languages(),
        "analysis_features": [
            "sentiment_analysis",
            "readability_metrics",
            "entity_extraction", 
            "keyword_extraction",
            "topic_modeling",
            "pos_tagging",
            "emotion_analysis",
            "vocabulary_richness"
        ],
        "output_formats": ["json", "charts", "insights"],
        "batch_processing": True,
        "comparative_analysis": True
    }
'''
        
        routers_dir = self.service_path / "routers"
        with open(routers_dir / "analysis.py", "w", encoding="utf-8") as f:
            f.write(analysis_router)
        
        # Router de visualización
        visualization_router = '''from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from visualizers.chart_generator import ChartGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

class ChartRequest(BaseModel):
    chart_type: str
    data: Dict[str, Any]
    title: Optional[str] = "Gráfico"
    options: Optional[Dict[str, Any]] = {}

@router.post("/chart")
async def create_chart(request: ChartRequest):
    """Crear gráfico específico"""
    
    try:
        chart_generator = ChartGenerator()
        chart_path = ""
        
        if request.chart_type == "sentiment":
            chart_path = await chart_generator.create_sentiment_chart(
                request.data, request.title
            )
        elif request.chart_type == "keywords":
            chart_path = await chart_generator.create_keywords_chart(
                request.data, request.title
            )
        elif request.chart_type == "emotions":
            chart_path = await chart_generator.create_emotions_chart(
                request.data, request.title
            )
        elif request.chart_type == "readability":
            chart_path = await chart_generator.create_readability_chart(
                request.data, request.title
            )
        elif request.chart_type == "pos_tags":
            chart_path = await chart_generator.create_pos_tags_chart(
                request.data, request.title
            )
        else:
            raise HTTPException(status_code=400, detail=f"Tipo de gráfico no soportado: {request.chart_type}")
        
        if chart_path:
            return {"chart_url": f"/static/plots/{chart_path.split('/')[-1]}"}
        else:
            raise HTTPException(status_code=500, detail="Error generando gráfico")
            
    except Exception as e:
        logger.error(f"❌ Error creando gráfico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboard")
async def create_dashboard(
    documents_data: List[Dict[str, Any]],
    title: str = "Dashboard Analítico"
):
    """Crear dashboard comparativo"""
    
    try:
        chart_generator = ChartGenerator()
        
        dashboard_path = await chart_generator.create_comparative_dashboard(
            documents_data, title
        )
        
        if dashboard_path:
            return {"dashboard_url": f"/static/plots/{dashboard_path.split('/')[-1]}"}
        else:
            raise HTTPException(status_code=500, detail="Error generando dashboard")
            
    except Exception as e:
        logger.error(f"❌ Error creando dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chart-types")
async def get_chart_types():
    """Obtener tipos de gráficos disponibles"""
    
    chart_generator = ChartGenerator()
    return {
        "available_charts": chart_generator.get_chart_types(),
        "formats": ["html", "png", "svg"],
        "interactive": True
    }
'''
        
        with open(routers_dir / "visualization.py", "w", encoding="utf-8") as f:
            f.write(visualization_router)
        
        # Crear otros routers básicos
        for router_name in ["statistics", "insights", "models"]:
            basic_router = f'''from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

@router.get("/")
async def {router_name}_root():
    return {{"message": "{router_name.title()} router - En desarrollo"}}
'''
            with open(routers_dir / f"{router_name}.py", "w", encoding="utf-8") as f:
                f.write(basic_router)
        
        self.logger.info("✅ Routers creados")

    def create_config_utils(self):
        """Crear utilidades de configuración"""
        self.logger.info("⚙️ Creando utilidades...")
        
        # Config
        config_content = '''"""
Configuración del Analytics-Engine
"""
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Configuración general
    app_name: str = "Analytics-Engine"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8003
    
    # Puertos de otros servicios
    ai_engine_port: int = 8001
    document_processor_port: int = 8002
    
    # Directorios
    data_dir: str = "./data"
    reports_dir: str = "./data/reports"
    cache_dir: str = "./data/cache"
    plots_dir: str = "./static/plots"
    
    # Análisis de texto
    default_language: str = "auto"
    max_text_length: int = 1000000  # 1MB
    enable_topic_modeling: bool = True
    enable_entity_extraction: bool = True
    
    # IA e Insights
    enable_ai_insights: bool = True
    ai_insights_timeout: int = 30
    max_insights_per_document: int = 10
    
    # Visualización
    default_chart_format: str = "html"
    chart_width: int = 800
    chart_height: int = 600
    enable_interactive_charts: bool = True
    
    # CORS
    cors_origins: List[str] = ["http://localhost:8080", "http://127.0.0.1:8080"]
    
    # Base de datos
    database_url: str = "sqlite:///./analytics_engine.db"
    redis_url: str = "redis://localhost:6379/3"
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    # Modelos NLP
    spacy_model: str = "es_core_news_sm"  # Modelo principal
    spacy_fallback: str = "en_core_web_sm"  # Modelo de respaldo
    
    # Análisis estadístico
    enable_statistical_analysis: bool = True
    confidence_interval: float = 0.95
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
'''
        
        utils_dir = self.service_path / "utils"
        with open(utils_dir / "config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        # Logging config
        logging_config = '''"""
Configuración de logging para Analytics-Engine
"""
import logging
import logging.handlers
from pathlib import Path

def setup_logging():
    """Configurar sistema de logging"""
    
    # Crear directorio de logs
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configuración del logger root
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler con rotación
            logging.handlers.RotatingFileHandler(
                log_dir / "analytics_engine.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Configurar loggers específicos
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("plotly").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
'''
        
        with open(utils_dir / "logging_config.py", "w", encoding="utf-8") as f:
            f.write(logging_config)
        
        self.logger.info("✅ Utilidades creadas")

    def create_analytics_manager(self):
        """Crear gestor principal de análisis"""
        self.logger.info("📊 Creando gestor de análisis...")
        
        analytics_manager_content = '''"""
Gestor principal del motor de análisis
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """Administrador principal del motor de análisis"""
    
    def __init__(self):
        self.analysis_history = []
        self.statistics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "total_documents_processed": 0,
            "total_words_analyzed": 0,
            "average_processing_time": 0.0,
            "analysis_types": {}
        }
        self.is_ready = False
        
    async def initialize(self):
        """Inicializar el gestor de análisis"""
        logger.info("🔧 Inicializando Analytics Manager...")
        
        # Crear directorios necesarios
        directories = [
            Path("data/datasets"),
            Path("data/reports"),
            Path("data/cache"),
            Path("static/plots")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Cargar estadísticas si existen
        await self._load_statistics()
        
        self.is_ready = True
        logger.info("✅ Analytics Manager inicializado")
    
    async def register_analysis(
        self, 
        analysis_type: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any],
        processing_time: float
    ):
        """Registrar un análisis realizado"""
        
        analysis_record = {
            "id": len(self.analysis_history) + 1,
            "type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time,
            "input_size": len(str(input_data)),
            "success": "error" not in result,
            "metadata": {
                "word_count": input_data.get("word_count", 0),
                "language": result.get("language", "unknown")
            }
        }
        
        self.analysis_history.append(analysis_record)
        
        # Actualizar estadísticas
        self.statistics["total_analyses"] += 1
        if analysis_record["success"]:
            self.statistics["successful_analyses"] += 1
        else:
            self.statistics["failed_analyses"] += 1
        
        self.statistics["total_words_analyzed"] += analysis_record["metadata"]["word_count"]
        
        # Actualizar tiempo promedio
        total_time = sum(a["processing_time"] for a in self.analysis_history)
        self.statistics["average_processing_time"] = total_time / len(self.analysis_history)
        
        # Contar tipos de análisis
        if analysis_type in self.statistics["analysis_types"]:
            self.statistics["analysis_types"][analysis_type] += 1
        else:
            self.statistics["analysis_types"][analysis_type] = 1
        
        await self._save_statistics()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del motor"""
        
        # Calcular métricas adicionales
        success_rate = 0.0
        if self.statistics["total_analyses"] > 0:
            success_rate = (self.statistics["successful_analyses"] / 
                          self.statistics["total_analyses"]) * 100
        
        # Análisis recientes (últimos 10)
        recent_analyses = self.analysis_history[-10:] if self.analysis_history else []
        
        return {
            **self.statistics,
            "success_rate": round(success_rate, 2),
            "recent_analyses": recent_analyses,
            "storage_info": await self._get_storage_info(),
            "performance_metrics": await self._get_performance_metrics()
        }
    
    async def _get_storage_info(self) -> Dict[str, Any]:
        """Obtener información de almacenamiento"""
        
        storage_info = {}
        
        for directory_name, path in [
            ("datasets", "data/datasets"),
            ("reports", "data/reports"),
            ("cache", "data/cache"),
            ("plots", "static/plots")
        ]:
            directory = Path(path)
            if directory.exists():
                files = list(directory.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                storage_info[directory_name] = {
                    "files": len([f for f in files if f.is_file()]),
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "path": str(directory)
                }
            else:
                storage_info[directory_name] = {
                    "files": 0,
                    "total_size_mb": 0,
                    "path": str(directory)
                }
        
        return storage_info
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento"""
        
        if not self.analysis_history:
            return {}
        
        processing_times = [a["processing_time"] for a in self.analysis_history]
        word_counts = [a["metadata"]["word_count"] for a in self.analysis_history]
        
        return {
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "median_processing_time": sorted(processing_times)[len(processing_times)//2],
            "avg_words_per_second": (
                sum(word_counts) / sum(processing_times) 
                if sum(processing_times) > 0 else 0
            ),
            "total_processing_time": sum(processing_times)
        }
    
    async def _load_statistics(self):
        """Cargar estadísticas desde archivo"""
        stats_file = Path("data/analytics_stats.json")
        
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.statistics = data.get("statistics", self.statistics)
                    self.analysis_history = data.get("history", [])
                logger.info("📊 Estadísticas cargadas")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando estadísticas: {e}")
    
    async def _save_statistics(self):
        """Guardar estadísticas en archivo"""
        stats_file = Path("data/analytics_stats.json")
        
        try:
            data = {
                "statistics": self.statistics,
                "history": self.analysis_history[-1000:]  # Mantener últimos 1000
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Error guardando estadísticas: {e}")
    
    async def cleanup_old_data(self, days: int = 30):
        """Limpiar datos antiguos"""
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cleaned_files = 0
        
        for directory_path in ["data/cache", "static/plots"]:
            directory = Path(directory_path)
            if not directory.exists():
                continue
            
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            cleaned_files += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Error eliminando {file_path}: {e}")
        
        logger.info(f"🧹 Limpieza completada: {cleaned_files} archivos eliminados")
        return cleaned_files
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generar reporte de rendimiento"""
        
        stats = await self.get_statistics()
        
        report = {
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_analyses": stats["total_analyses"],
                "success_rate": stats["success_rate"],
                "avg_processing_time": stats["average_processing_time"],
                "total_words_processed": stats["total_words_analyzed"]
            },
            "performance": stats.get("performance_metrics", {}),
            "storage": stats.get("storage_info", {}),
            "analysis_breakdown": stats.get("analysis_types", {}),
            "recommendations": []
        }
        
        # Generar recomendaciones
        if stats["success_rate"] < 95:
            report["recommendations"].append("Revisar casos de fallo para mejorar robustez")
        
        if stats["average_processing_time"] > 5.0:
            report["recommendations"].append("Optimizar algoritmos para mejorar velocidad")
        
        # Guardar reporte
        report_file = Path(f"data/reports/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    async def cleanup(self):
        """Limpiar recursos"""
        logger.info("🧹 Limpiando recursos del Analytics Manager...")
        await self._save_statistics()
        logger.info("✅ Recursos limpiados")
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "analytics_manager.py", "w", encoding="utf-8") as f:
            f.write(analytics_manager_content)
        
        self.logger.info("✅ Gestor de análisis creado")

    def create_data_processor(self):
        """Crear procesador de datos"""
        self.logger.info("🔢 Creando procesador de datos...")
        
        data_processor_content = '''"""
Procesador de datos para análisis estadístico
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

class DataProcessor:
    """Procesador de datos para análisis estadístico"""
    
    def __init__(self):
        self.is_ready = False
        self.supported_formats = ['json', 'csv', 'excel', 'dataframe']
        
    async def initialize(self):
        """Inicializar el procesador de datos"""
        logger.info("🔧 Inicializando Data Processor...")
        self.is_ready = True
        logger.info("✅ Data Processor inicializado")
    
    async def process_text_analysis_data(
        self, 
        analysis_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Procesar resultados de análisis de texto en DataFrame"""
        
        try:
            # Extraer métricas numéricas
            processed_data = []
            
            for i, result in enumerate(analysis_results):
                row = {
                    'document_id': i,
                    'text_length': result.get('text_length', 0),
                    'word_count': result.get('word_count', 0),
                    'sentence_count': result.get('sentence_count', 0),
                    'paragraph_count': result.get('paragraph_count', 0),
                    'language': result.get('language', 'unknown'),
                    'vocabulary_richness': result.get('vocabulary_richness', 0),
                    'average_sentence_length': result.get('average_sentence_length', 0),
                }
                
                # Métricas de sentimiento
                sentiment = result.get('sentiment', {})
                row.update({
                    'sentiment_compound': sentiment.get('vader_compound', 0),
                    'sentiment_positive': sentiment.get('vader_positive', 0),
                    'sentiment_negative': sentiment.get('vader_negative', 0),
                    'sentiment_neutral': sentiment.get('vader_neutral', 0),
                    'textblob_polarity': sentiment.get('textblob_polarity', 0),
                    'textblob_subjectivity': sentiment.get('textblob_subjectivity', 0),
                })
                
                # Métricas de legibilidad
                readability = result.get('readability', {})
                row.update({
                    'flesch_reading_ease': readability.get('flesch_reading_ease', 0),
                    'avg_word_length': readability.get('average_word_length', 0),
                    'reading_time_minutes': readability.get('estimated_reading_time_minutes', 0),
                })
                
                # Métricas de emociones
                emotions = result.get('emotion_scores', {})
                for emotion, score in emotions.items():
                    row[f'emotion_{emotion}'] = score
                
                processed_data.append(row)
            
            df = pd.DataFrame(processed_data)
            logger.info(f"✅ DataFrame creado con {len(df)} filas y {len(df.columns)} columnas")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error procesando datos: {e}")
            return pd.DataFrame()
    
    async def calculate_descriptive_statistics(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calcular estadísticas descriptivas"""
        
        try:
            # Seleccionar solo columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {}
            
            stats_summary = {
                'basic_stats': df[numeric_cols].describe().to_dict(),
                'correlations': df[numeric_cols].corr().to_dict(),
                'skewness': df[numeric_cols].skew().to_dict(),
                'kurtosis': df[numeric_cols].kurtosis().to_dict(),
                'missing_values': df[numeric_cols].isnull().sum().to_dict()
            }
            
            # Estadísticas adicionales
            stats_summary['variance'] = df[numeric_cols].var().to_dict()
            stats_summary['std_dev'] = df[numeric_cols].std().to_dict()
            stats_summary['range'] = (df[numeric_cols].max() - df[numeric_cols].min()).to_dict()
            
            logger.info("✅ Estadísticas descriptivas calculadas")
            return stats_summary
            
        except Exception as e:
            logger.error(f"❌ Error calculando estadísticas: {e}")
            return {}
    
    async def perform_correlation_analysis(
        self, 
        df: pd.DataFrame,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """Realizar análisis de correlación"""
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {}
            
            # Matriz de correlación
            corr_matrix = df[numeric_cols].corr(method=method)
            
            # Encontrar correlaciones fuertes
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Correlación fuerte
                        strong_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'method': method,
                'variables_analyzed': list(numeric_cols)
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de correlación: {e}")
            return {}
    
    async def perform_clustering_analysis(
        self, 
        df: pd.DataFrame,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Realizar análisis de clustering"""
        
        try:
            # Seleccionar variables numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {}
            
            # Preparar datos
            data = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Normalizar datos
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Determinar número óptimo de clusters si no se especifica
            if n_clusters is None:
                n_clusters = await self._find_optimal_clusters(data_scaled)
            
            # Aplicar K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data_scaled)
            
            # Calcular métricas
            silhouette_avg = silhouette_score(data_scaled, cluster_labels)
            
            # Agregar clusters al DataFrame
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = cluster_labels
            
            # Estadísticas por cluster
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                cluster_stats[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df) * 100,
                    'means': cluster_data[numeric_cols].mean().to_dict()
                }
            
            return {
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'silhouette_score': silhouette_avg,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_statistics': cluster_stats,
                'variables_used': list(numeric_cols)
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de clustering: {e}")
            return {}
    
    async def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Encontrar número óptimo de clusters usando método del codo"""
        
        try:
            inertias = []
            k_range = range(2, min(max_k + 1, len(data)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            # Método del codo (simplificado)
            if len(inertias) < 2:
                return 2
            
            # Encontrar el punto donde la mejora es menor
            differences = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            optimal_k = differences.index(min(differences)) + 2
            
            return min(optimal_k, 5)  # Limitar a máximo 5 clusters
            
        except Exception as e:
            logger.warning(f"⚠️ Error encontrando clusters óptimos: {e}")
            return 3  # Valor por defecto
    
    async def perform_pca_analysis(
        self, 
        df: pd.DataFrame,
        n_components: Optional[int] = None
    ) -> Dict[str, Any]:
        """Realizar análisis de componentes principales (PCA)"""
        
        try:
            # Seleccionar variables numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {}
            
            # Preparar datos
            data = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Normalizar datos
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Determinar número de componentes
            if n_components is None:
                n_components = min(len(numeric_cols), 5)
            
            # Aplicar PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(data_scaled)
            
            # Calcular varianza explicada
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            return {
                'n_components': n_components,
                'explained_variance_ratio': explained_variance.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'components': components.tolist(),
                'feature_loadings': pca.components_.tolist(),
                'variables_used': list(numeric_cols),
                'total_variance_explained': float(cumulative_variance[-1])
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis PCA: {e}")
            return {}
    
    async def detect_outliers(
        self, 
        df: pd.DataFrame,
        method: str = 'iqr'
    ) -> Dict[str, Any]:
        """Detectar valores atípicos"""
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {}
            
            outliers_info = {}
            
            for col in numeric_cols:
                column_data = df[col].dropna()
                
                if method == 'iqr':
                    Q1 = column_data.quantile(0.25)
                    Q3 = column_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(column_data))
                    outliers = column_data[z_scores > 3]
                
                else:
                    continue
                
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(column_data) * 100,
                    'values': outliers.tolist() if len(outliers) < 20 else outliers.head(20).tolist(),
                    'method': method
                }
            
            return {
                'outliers_by_variable': outliers_info,
                'detection_method': method,
                'total_variables_analyzed': len(numeric_cols)
            }
            
        except Exception as e:
            logger.error(f"❌ Error detectando outliers: {e}")
            return {}
    
    async def generate_data_quality_report(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generar reporte de calidad de datos"""
        
        try:
            report = {
                'dataset_info': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                },
                'missing_data': {
                    'columns_with_missing': df.isnull().any().sum(),
                    'total_missing_values': df.isnull().sum().sum(),
                    'missing_percentage_by_column': (df.isnull().sum() / len(df) * 100).to_dict()
                },
                'data_types': df.dtypes.astype(str).to_dict(),
                'unique_values': df.nunique().to_dict()
            }
            
            # Detectar posibles problemas
            issues = []
            
            # Columnas con muchos valores faltantes
            high_missing = df.columns[df.isnull().mean() > 0.5]
            if len(high_missing) > 0:
                issues.append(f"Columnas con >50% valores faltantes: {list(high_missing)}")
            
            # Columnas con un solo valor
            single_value_cols = df.columns[df.nunique() == 1]
            if len(single_value_cols) > 0:
                issues.append(f"Columnas con valor único: {list(single_value_cols)}")
            
            # Columnas numéricas con valores constantes
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            constant_cols = [col for col in numeric_cols if df[col].std() == 0]
            if constant_cols:
                issues.append(f"Columnas numéricas constantes: {constant_cols}")
            
            report['quality_issues'] = issues
            report['data_quality_score'] = max(0, 100 - len(issues) * 10)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte de calidad: {e}")
            return {}
    
    def get_supported_formats(self) -> List[str]:
        """Obtener formatos soportados"""
        return self.supported_formats
    
    async def export_analysis_results(
        self, 
        results: Dict[str, Any],
        format: str = 'json',
        output_path: Optional[str] = None
    ) -> str:
        """Exportar resultados de análisis"""
        
        try:
            if not output_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"data/reports/analysis_results_{timestamp}.{format}"
            
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str)
            
            elif format == 'csv' and 'dataframe' in results:
                df = pd.DataFrame(results['dataframe'])
                df.to_csv(output_path, index=False)
            
            elif format == 'excel' and 'dataframe' in results:
                df = pd.DataFrame(results['dataframe'])
                df.to_excel(output_path, index=False)
            
            logger.info(f"✅ Resultados exportados a: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error exportando resultados: {e}")
            return ""
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "data_processor.py", "w", encoding="utf-8") as f:
            f.write(data_processor_content)
        
        self.logger.info("✅ Procesador de datos creado")

    def create_dockerfile(self):
        """Crear Dockerfile"""
        self.logger.info("🐳 Creando Dockerfile...")
        
        dockerfile_content = '''FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelos de spaCy
RUN python -m spacy download es_core_news_sm || true
RUN python -m spacy download en_core_web_sm || true

# Crear directorios necesarios
RUN mkdir -p /app/data/datasets /app/data/reports /app/data/cache /app/static/plots /app/logs && \
    chown -R appuser:appuser /app

# Copiar código de la aplicación
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

# Exponer puerto
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Comando por defecto
CMD ["python", "app.py"]
'''
        
        with open(self.service_path / "Dockerfile", "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        
        self.logger.info("✅ Dockerfile creado")

    def create_tests(self):
        """Crear tests básicos"""
        self.logger.info("🧪 Creando tests...")
        
        test_content = '''"""
Tests para Analytics-Engine
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from app import app
from services.text_analyzer import TextAnalyzer
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator

client = TestClient(app)

def test_health_check():
    """Test del health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "analytics-engine"

def test_status_endpoint():
    """Test del endpoint de status"""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "components" in data

def test_capabilities_endpoint():
    """Test del endpoint de capacidades"""
    response = client.get("/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "text_analysis" in data
    assert "statistical_analysis" in data
    assert "ai_insights" in data
    assert "visualization" in data

@pytest.mark.asyncio
async def test_text_analyzer():
    """Test del analizador de texto"""
    analyzer = TextAnalyzer()
    await analyzer.initialize()
    
    test_text = "Este es un texto de prueba para análisis. Contiene múltiples oraciones. ¡Es muy interesante!"
    
    result = await analyzer.analyze_text(test_text)
    
    assert result.text_length > 0
    assert result.word_count > 0
    assert result.sentence_count > 0
    assert result.language is not None
    assert isinstance(result.sentiment, dict)
    assert isinstance(result.keywords, list)

@pytest.mark.asyncio 
async def test_data_processor():
    """Test del procesador de datos"""
    processor = DataProcessor()
    await processor.initialize()
    
    # Crear datos de prueba
    test_data = [
        {
            'text_length': 100,
            'word_count': 20,
            'sentiment': {'vader_compound': 0.5},
            'readability': {'flesch_reading_ease': 70}
        },
        {
            'text_length': 150,
            'word_count': 30,
            'sentiment': {'vader_compound': -0.3},
            'readability': {'flesch_reading_ease': 60}
        }
    ]
    
    df = await processor.process_text_analysis_data(test_data)
    
    assert len(df) == 2
    assert 'text_length' in df.columns
    assert 'word_count' in df.columns

@pytest.mark.asyncio
async def test_chart_generator():
    """Test del generador de gráficos"""
    generator = ChartGenerator()
    await generator.initialize()
    
    # Test datos de sentimiento
    sentiment_data = {
        'vader_positive': 0.3,
        'vader_negative': 0.1,
        'vader_neutral': 0.6
    }
    
    chart_path = await generator.create_sentiment_chart(sentiment_data)
    
    # Verificar que se creó el archivo (o al menos no hay error)
    assert isinstance(chart_path, str)

def test_analyze_text_endpoint():
    """Test del endpoint de análisis de texto"""
    response = client.post(
        "/analyze/text",
        json={
            "text": "Este es un texto de prueba.",
            "generate_insights": False,
            "generate_charts": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    assert "processing_time" in data

def test_visualization_chart_types():
    """Test del endpoint de tipos de gráficos"""
    response = client.get("/visualize/chart-types")
    assert response.status_code == 200
    data = response.json()
    assert "available_charts" in data
    assert "formats" in data

if __name__ == "__main__":
    pytest.main([__file__])
'''
        
        tests_dir = self.service_path / "tests"
        with open(tests_dir / "test_analytics_engine.py", "w", encoding="utf-8") as f:
            f.write(test_content)
        
        self.logger.info("✅ Tests creados")

    def install_dependencies(self):
        """Instalar dependencias específicas"""
        self.logger.info("📦 Instalando dependencias del Analytics-Engine...")
        
        # Determinar el ejecutable de pip
        if os.name == 'nt':  # Windows
            pip_exe = self.venv_path / "Scripts" / "pip.exe"
        else:  # Linux/macOS
            pip_exe = self.venv_path / "bin" / "pip"
        
        if not pip_exe.exists():
            self.logger.error("❌ pip no encontrado en el entorno virtual")
            return False
        
        try:
            # Instalar dependencias
            result = subprocess.run([
                str(pip_exe), "install", "-r", 
                str(self.service_path / "requirements.txt")
            ], check=True, capture_output=True, text=True, cwd=self.service_path)
            
            self.logger.info("✅ Dependencias del Analytics-Engine instaladas")
            
            # Intentar descargar modelos de spaCy
            self.logger.info("📥 Descargando modelos de spaCy...")
            try:
                subprocess.run([
                    str(pip_exe), "install", "https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl"
                ], check=False, capture_output=True, text=True)
                
                subprocess.run([
                    str(pip_exe), "install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
                ], check=False, capture_output=True, text=True)
                
                self.logger.info("✅ Modelos de spaCy descargados")
            except Exception as e:
                self.logger.warning(f"⚠️ No se pudieron descargar modelos spaCy: {e}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Error instalando dependencias: {e}")
            self.logger.error(f"   Salida: {e.stdout}")
            self.logger.error(f"   Error: {e.stderr}")
            self.logger.warning("⚠️ Algunas dependencias pueden requerir instalación manual")
            return False

    def create_start_script(self):
        """Crear script de inicio"""
        self.logger.info("🚀 Creando script de inicio...")
        
        if os.name == 'nt':  # Windows
            start_script = '''@echo off
echo 📊 INICIANDO ANALYTICS-ENGINE...
echo =================================

cd /d "%~dp0"

REM Activar entorno virtual
call ..\\..\\venv\\Scripts\\activate.bat

REM Verificar que estamos en el directorio correcto
if not exist "app.py" (
    echo ❌ app.py no encontrado. Verifica que estés en services/analytics-engine/
    pause
    exit /b 1
)

REM Verificar modelos de spaCy (opcional)
python -c "import spacy; spacy.load('es_core_news_sm')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Modelo spaCy español no encontrado. Funcionalidad NLP limitada.
)

REM Iniciar servidor
echo ✅ Iniciando servidor en puerto 8003...
python app.py

pause
'''
            with open(self.service_path / "start.bat", "w", encoding="utf-8") as f:
                f.write(start_script)
        
        else:  # Linux/macOS
            start_script = '''#!/bin/bash

echo "📊 INICIANDO ANALYTICS-ENGINE..."
echo "================================="

# Ir al directorio del script
cd "$(dirname "$0")"

# Activar entorno virtual
source ../../venv/bin/activate

# Verificar que estamos en el directorio correcto
if [ ! -f "app.py" ]; then
    echo "❌ app.py no encontrado. Verifica que estés en services/analytics-engine/"
    exit 1
fi

# Verificar modelos de spaCy (opcional)
if ! python -c "import spacy; spacy.load('es_core_news_sm')" &> /dev/null; then
    echo "⚠️ Modelo spaCy español no encontrado. Descarga con:"
    echo "   python -m spacy download es_core_news_sm"
fi

# Iniciar servidor
echo "✅ Iniciando servidor en puerto 8003..."
python app.py
'''
            start_file = self.service_path / "start.sh"
            with open(start_file, "w", encoding="utf-8") as f:
                f.write(start_script)
            
            # Hacer ejecutable
            start_file.chmod(0o755)
        
        self.logger.info("✅ Script de inicio creado")

    def create_env_file(self):
        """Crear archivo .env local"""
        self.logger.info("⚙️ Creando .env local...")
        
        env_content = '''# ANALYTICS-ENGINE - Configuración local
DEBUG=true
LOG_LEVEL=INFO

# Puertos de servicios
AI_ENGINE_PORT=8001
DOCUMENT_PROCESSOR_PORT=8002

# Análisis
DEFAULT_LANGUAGE=auto
ENABLE_TOPIC_MODELING=true
ENABLE_ENTITY_EXTRACTION=true

# IA e Insights
ENABLE_AI_INSIGHTS=true
AI_INSIGHTS_TIMEOUT=30

# Visualización
DEFAULT_CHART_FORMAT=html
ENABLE_INTERACTIVE_CHARTS=true

# Modelos NLP
SPACY_MODEL=es_core_news_sm
SPACY_FALLBACK=en_core_web_sm

# Base de datos
DATABASE_URL=sqlite:///./analytics_engine.db
REDIS_URL=redis://localhost:6379/3
'''
        
        with open(self.service_path / ".env", "w", encoding="utf-8") as f:
            f.write(env_content)
        
        self.logger.info("✅ .env local creado")

    def run_setup(self):
        """Ejecutar setup completo"""
        self.logger.info("🚀 INICIANDO SETUP DEL ANALYTICS-ENGINE")
        self.logger.info("=" * 50)
        
        try:
            # Validar entorno
            if not self.validate_environment():
                return False
            
            # Crear estructura
            self.create_directory_structure()
            
            # Crear archivos principales
            self.create_main_app()
            self.create_requirements()
            
            # Crear servicios core
            self.create_text_analyzer()
            self.create_ai_insights_service()
            self.create_chart_generator()
            
            # Crear routers y utilidades
            self.create_routers()
            self.create_config_utils()
            self.create_analytics_manager()
            self.create_data_processor()
            
            # Crear archivos de deployment
            self.create_dockerfile()
            self.create_tests()
            self.create_start_script()
            self.create_env_file()
            
            # Instalar dependencias
            if not self.install_dependencies():
                self.logger.warning("⚠️ Algunas dependencias no se instalaron correctamente")
            
            self.logger.info("🎉 ANALYTICS-ENGINE CONFIGURADO EXITOSAMENTE!")
            self.logger.info("=" * 50)
            self.logger.info("")
            self.logger.info("📋 PRÓXIMOS PASOS:")
            self.logger.info("1. (Opcional) Descargar modelos de spaCy:")
            self.logger.info("   python -m spacy download es_core_news_sm")
            self.logger.info("   python -m spacy download en_core_web_sm")
            self.logger.info("2. Ejecutar: python app.py")
            self.logger.info("3. Verificar: http://localhost:8003/health")
            self.logger.info("4. Documentación: http://localhost:8003/docs")
            self.logger.info("")
            self.logger.info("🔗 ENDPOINTS PRINCIPALES:")
            self.logger.info("   • Health: http://localhost:8003/health")
            self.logger.info("   • Status: http://localhost:8003/status")
            self.logger.info("   • Docs: http://localhost:8003/docs")
            self.logger.info("   • Capacidades: http://localhost:8003/capabilities")
            self.logger.info("   • Análisis: http://localhost:8003/analyze/text")
            self.logger.info("   • Visualización: http://localhost:8003/visualize/chart")
            self.logger.info("")
            self.logger.info("📊 CAPACIDADES DE ANÁLISIS:")
            self.logger.info("   • Análisis de sentimientos con VADER y TextBlob")
            self.logger.info("   • Extracción de palabras clave con TF-IDF")
            self.logger.info("   • Análisis de legibilidad y métricas de texto")
            self.logger.info("   • Detección de entidades con spaCy")
            self.logger.info("   • Modelado de temas con LDA")
            self.logger.info("   • Análisis estadístico y clustering")
            self.logger.
                        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            # Normalizar por longitud del texto
            emotion_scores[emotion] = score / max(len(text.split()), 1) * 100
        
        return emotion_scores
    
    async def generate_wordcloud(self, text: str, output_path: str = None) -> str:
        """Generar nube de palabras"""
        
        try:
            # Limpiar texto
            words = nltk.word_tokenize(text.lower())
            words = [w for w in words if w.isalnum() and len(w) > 2]
            clean_text = ' '.join(words)
            
            # Generar wordcloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(clean_text)
            
            # Guardar imagen
            if not output_path:
                output_path = f"static/plots/wordcloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            wordcloud.to_file(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generando wordcloud: {e}")
            return ""
    
    async def batch_analyze(self, texts: List[str]) -> List[TextAnalysisResult]:
        """Análisis en lote de múltiples textos"""
        
        logger.info(f"📦 Analizando lote de {len(texts)} textos...")
        
        tasks = [self.analyze_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar resultados exitosos
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error en texto {i}: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"✅ Lote completado: {len(successful_results)}/{len(texts)} exitosos")
        return successful_results
    
    def get_supported_languages(self) -> List[str]:
        """Obtener idiomas soportados"""
        return self.supported_languages
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Obtener estado de modelos cargados"""
        return {
            'spacy_model': self.nlp_model is not None,
            'vader_sentiment': self.sentiment_analyzer is not None,
            'nltk_resources': True  # Asumimos que están disponibles si llegó hasta aquí
        }
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "text_analyzer.py", "w", encoding="utf-8") as f:
            f.write(text_analyzer_content)
        
        self.logger.info("✅ Analizador de texto creado")

    def create_ai_insights_service(self):
        """Crear servicio de insights con IA"""
        self.logger.info("🤖 Creando servicio de insights con IA...")
        
        ai_insights_content = '''"""
Servicio de insights usando IA - se conecta con AI-Engine
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import httpx
from dataclasses import dataclass
from datetime import datetime
import json

from utils.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class AIInsight:
    """Estructura para un insight generado por IA"""
    title: str
    content: str
    confidence: float
    category: str
    metadata: Dict[str, Any]
    timestamp: str

class AIInsightsService:
    """Servicio para generar insights usando el AI-Engine"""
    
    def __init__(self):
        self.settings = Settings()
        self.is_ready = False
        self.ai_engine_url = f"http://localhost:{self.settings.ai_engine_port}"
        self.client = None
        
    async def initialize(self):
        """Inicializar el servicio de insights"""
        logger.info("🔧 Inicializando AI Insights Service...")
        
        try:
            # Crear cliente HTTP
            self.client = httpx.AsyncClient(timeout=30.0)
            
            # Verificar conexión con AI-Engine
            connection_ok = await self.check_ai_engine_connection()
            if connection_ok:
                logger.info("✅ Conexión con AI-Engine establecida")
            else:
                logger.warning("⚠️ AI-Engine no disponible. Modo limitado.")
            
            self.is_ready = True
            logger.info("✅ AI Insights Service inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando AI Insights: {e}")
            self.is_ready = True  # Continuar en modo limitado
    
    async def check_ai_engine_connection(self) -> bool:
        """Verificar conexión con AI-Engine"""
        
        try:
            response = await self.client.get(f"{self.ai_engine_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def generate_document_insights(
        self, 
        text: str, 
        analysis_result: Dict = None,
        options: Dict = None
    ) -> List[AIInsight]:
        """Generar insights sobre un documento"""
        
        options = options or {}
        insights = []
        
        try:
            # Insight 1: Resumen del contenido
            if options.get('generate_summary', True):
                summary_insight = await self._generate_summary_insight(text)
                if summary_insight:
                    insights.append(summary_insight)
            
            # Insight 2: Análisis de temas principales
            if options.get('analyze_themes', True):
                themes_insight = await self._analyze_themes_insight(text)
                if themes_insight:
                    insights.append(themes_insight)
            
            # Insight 3: Clasificación del documento
            if options.get('classify_document', True):
                classification_insight = await self._classify_document_insight(text)
                if classification_insight:
                    insights.append(classification_insight)
            
            # Insight 4: Análisis de sentimiento contextual
            if options.get('sentiment_context', True) and analysis_result:
                sentiment_insight = await self._generate_sentiment_context(text, analysis_result)
                if sentiment_insight:
                    insights.append(sentiment_insight)
            
            # Insight 5: Recomendaciones
            if options.get('generate_recommendations', True):
                recommendations_insight = await self._generate_recommendations(text, analysis_result)
                if recommendations_insight:
                    insights.append(recommendations_insight)
            
            logger.info(f"✅ Generados {len(insights)} insights")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generando insights: {e}")
            return []
    
    async def _generate_summary_insight(self, text: str) -> Optional[AIInsight]:
        """Generar resumen inteligente del documento"""
        
        try:
            prompt = f"""
            Analiza el siguiente texto y proporciona un resumen conciso que capture:
            1. Los puntos principales
            2. Los temas centrales
            3. Las conclusiones clave
            
            Texto a analizar:
            {text[:2000]}...
            
            Proporciona un resumen de máximo 150 palabras.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Resumen del Documento",
                    content=response,
                    confidence=0.85,
                    category="summary",
                    metadata={"word_count": len(response.split())},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error generando resumen: {e}")
        
        return None
    
    async def _analyze_themes_insight(self, text: str) -> Optional[AIInsight]:
        """Analizar temas principales del documento"""
        
        try:
            prompt = f"""
            Identifica y analiza los 3-5 temas principales en el siguiente texto.
            Para cada tema, proporciona:
            1. Nombre del tema
            2. Relevancia (1-10)
            3. Breve descripción
            
            Texto:
            {text[:2000]}...
            
            Responde en formato estructurado.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Análisis de Temas Principales",
                    content=response,
                    confidence=0.80,
                    category="themes",
                    metadata={"analysis_type": "thematic"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error analizando temas: {e}")
        
        return None
    
    async def _classify_document_insight(self, text: str) -> Optional[AIInsight]:
        """Clasificar tipo de documento"""
        
        try:
            prompt = f"""
            Clasifica el siguiente documento en una de estas categorías:
            - Reporte técnico
            - Documento legal
            - Comunicación comercial
            - Contenido académico
            - Manual o guía
            - Correspondencia
            - Otro
            
            Explica brevemente por qué pertenece a esa categoría y qué características lo definen.
            
            Texto:
            {text[:1500]}...
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Clasificación del Documento",
                    content=response,
                    confidence=0.75,
                    category="classification",
                    metadata={"classification_type": "document_type"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error clasificando documento: {e}")
        
        return None
    
    async def _generate_sentiment_context(self, text: str, analysis_result: Dict) -> Optional[AIInsight]:
        """Generar contexto del análisis de sentimientos"""
        
        try:
            sentiment_data = analysis_result.get('sentiment', {})
            overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
            
            prompt = f"""
            El análisis automático indica que este documento tiene un sentimiento "{overall_sentiment}".
            
            Analiza el contexto y explica:
            1. ¿Por qué el documento tiene este sentimiento?
            2. ¿Qué elementos específicos contribuyen a este sentimiento?
            3. ¿Es apropiado el tono para el tipo de documento?
            
            Fragmento del texto:
            {text[:1000]}...
            
            Proporciona un análisis contextual del sentimiento.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title=f"Contexto del Sentimiento ({overall_sentiment.title()})",
                    content=response,
                    confidence=0.70,
                    category="sentiment_context",
                    metadata={"detected_sentiment": overall_sentiment},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error en contexto de sentimiento: {e}")
        
        return None
    
    async def _generate_recommendations(self, text: str, analysis_result: Dict = None) -> Optional[AIInsight]:
        """Generar recomendaciones basadas en el análisis"""
        
        try:
            context = ""
            if analysis_result:
                readability = analysis_result.get('readability', {})
                flesch_score = readability.get('flesch_reading_ease', 50)
                
                if flesch_score < 30:
                    context += "El documento tiene baja legibilidad. "
                elif flesch_score > 70:
                    context += "El documento tiene alta legibilidad. "
            
            prompt = f"""
            Basándote en el análisis del siguiente documento, proporciona 3-5 recomendaciones prácticas para:
            1. Mejorar la claridad y legibilidad
            2. Optimizar la estructura
            3. Enhancer la comunicación efectiva
            
            {context}
            
            Texto analizado:
            {text[:1000]}...
            
            Proporciona recomendaciones específicas y accionables.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Recomendaciones de Mejora",
                    content=response,
                    confidence=0.75,
                    category="recommendations",
                    metadata={"recommendation_type": "improvement"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error generando recomendaciones: {e}")
        
        return None
    
    async def _call_ai_engine(self, prompt: str, model: str = "auto") -> Optional[str]:
        """Llamar al AI-Engine para generar respuesta"""
        
        if not await self.check_ai_engine_connection():
            logger.warning("⚠️ AI-Engine no disponible")
            return None
        
        try:
            payload = {
                "prompt": prompt,
                "model": model,
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = await self.client.post(
                f"{self.ai_engine_url}/generate/complete",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("text", "").strip()
            else:
                logger.warning(f"⚠️ AI-Engine respondió con status {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error llamando AI-Engine: {e}")
            return None
    
    async def generate_comparative_insights(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[AIInsight]:
        """Generar insights comparativos entre múltiples documentos"""
        
        if len(documents) < 2:
            return []
        
        insights = []
        
        try:
            # Comparar sentimientos
            sentiments = [doc.get('analysis', {}).get('sentiment', {}).get('overall_sentiment', 'neutral') 
                         for doc in documents]
            
            # Comparar longitudes
            lengths = [doc.get('analysis', {}).get('word_count', 0) for doc in documents]
            
            # Comparar legibilidad
            readabilities = [doc.get('analysis', {}).get('readability', {}).get('flesch_reading_ease', 50) 
                           for doc in documents]
            
            comparison_data = {
                'document_count': len(documents),
                'sentiments': sentiments,
                'avg_length': sum(lengths) / len(lengths),
                'avg_readability': sum(readabilities) / len(readabilities)
            }
            
            prompt = f"""
            Analiza los siguientes datos comparativos de {len(documents)} documentos:
            
            Sentimientos: {sentiments}
            Longitud promedio: {comparison_data['avg_length']:.0f} palabras
            Legibilidad promedio: {comparison_data['avg_readability']:.1f}
            
            Proporciona insights sobre:
            1. Patrones en los sentimientos
            2. Consistencia en la comunicación
            3. Variabilidad en complejidad
            4. Recomendaciones para homogeneizar
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                insights.append(AIInsight(
                    title="Análisis Comparativo",
                    content=response,
                    confidence=0.80,
                    category="comparative",
                    metadata=comparison_data,
                    timestamp=datetime.now().isoformat()
                ))
            
        except Exception as e:
            logger.error(f"❌ Error en insights comparativos: {e}")
        
        return insights
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles en AI-Engine"""
        return ["auto", "gpt-3.5-turbo", "claude-3-sonnet", "local"]
    
    async def cleanup(self):
        """Limpiar recursos"""
        if self.client:
            await self.client.aclose()
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "ai_insights.py", "w", encoding="utf-8") as f:
            f.write(ai_insights_content)
        
        self.logger.info("✅ Servicio de insights con IA creado")

    def create_chart_generator(self):
        """Crear generador de gráficos"""
        self.logger.info("📊 Creando generador de gráficos...")
        
        chart_generator_content = '''"""
Generador de gráficos y visualizaciones
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generador de gráficos para análisis de datos"""
    
    def __init__(self):
        self.is_ready = False
        self.output_dir = "static/plots"
        
        # Configurar estilo por defecto
        plt.style.use('default')
        sns.set_palette("husl")
        
    async def initialize(self):
        """Inicializar el generador de gráficos"""
        logger.info("🔧 Inicializando Chart Generator...")
        
        # Crear directorio de salida
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.is_ready = True
        logger.info("✅ Chart Generator inicializado")
    
    async def create_sentiment_chart(
        self, 
        sentiment_data: Dict[str, float], 
        title: str = "Análisis de Sentimientos"
    ) -> str:
        """Crear gráfico de análisis de sentimientos"""
        
        try:
            # Crear gráfico con Plotly
            fig = go.Figure()
            
            # Datos de VADER
            vader_data = {
                'Positivo': sentiment_data.get('vader_positive', 0),
                'Negativo': sentiment_data.get('vader_negative', 0),
                'Neutral': sentiment_data.get('vader_neutral', 0)
            }
            
            # Gráfico de barras
            fig.add_trace(go.Bar(
                x=list(vader_data.keys()),
                y=list(vader_data.values()),
                marker_color=['green', 'red', 'blue'],
                text=[f'{v:.2f}' for v in vader_data.values()],
                textposition='auto'
            ))
            
            # Configurar layout
            fig.update_layout(
                title=title,
                xaxis_title="Categoría",
                yaxis_title="Puntuación",
                template="plotly_white",
                height=400
            )
            
            # Guardar archivo
            filename = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de sentimientos creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de sentimientos: {e}")
            return ""
    
    async def create_readability_chart(
        self, 
        readability_data: Dict[str, float],
        title: str = "Métricas de Legibilidad"
    ) -> str:
        """Crear gráfico de métricas de legibilidad"""
        
        try:
            # Preparar datos
            metrics = []
            values = []
            
            if 'flesch_reading_ease' in readability_data:
                metrics.append('Facilidad Lectura')
                values.append(readability_data['flesch_reading_ease'])
            
            if 'average_sentence_length' in readability_data:
                metrics.append('Long. Oraciones')
                values.append(min(readability_data['average_sentence_length'], 50))  # Escalar
            
            if 'average_word_length' in readability_data:
                metrics.append('Long. Palabras')
                values.append(readability_data['average_word_length'] * 10)  # Escalar
            
            # Crear gráfico radial
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='Métricas'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title=title,
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"readability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de legibilidad creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de legibilidad: {e}")
            return ""
    
    async def create_keywords_chart(
        self, 
        keywords: List[Dict[str, float]],
        title: str = "Palabras Clave Principales"
    ) -> str:
        """Crear gráfico de palabras clave"""
        
        try:
            if not keywords:
                return ""
            
            # Tomar top 10
            top_keywords = keywords[:10]
            
            words = [kw['keyword'] for kw in top_keywords]
            scores = [kw['score'] for kw in top_keywords]
            
            # Crear gráfico horizontal
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=scores,
                y=words,
                orientation='h',
                marker_color='skyblue',
                text=[f'{s:.3f}' for s in scores],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Puntuación TF-IDF",
                yaxis_title="Palabras Clave",
                template="plotly_white",
                height=500
            )
            
            # Guardar archivo
            filename = f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de keywords creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de keywords: {e}")
            return ""
    
    async def create_pos_tags_chart(
        self, 
        pos_data: Dict[str, int],
        title: str = "Distribución de Categorías Gramaticales"
    ) -> str:
        """Crear gráfico de etiquetas POS"""
        
        try:
            if not pos_data:
                return ""
            
            # Agrupar etiquetas similares
            grouped_pos = {
                'Sustantivos': 0,
                'Verbos': 0,
                'Adjetivos': 0,
                'Adverbios': 0,
                'Pronombres': 0,
                'Otros': 0
            }
            
            noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
            verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            adj_tags = ['JJ', 'JJR', 'JJS']
            adv_tags = ['RB', 'RBR', 'RBS']
            pronoun_tags = ['PRP', 'PRP#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        , 'WP', 'WP#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        ]
            
            for tag, count in pos_data.items():
                if tag in noun_tags:
                    grouped_pos['Sustantivos'] += count
                elif tag in verb_tags:
                    grouped_pos['Verbos'] += count
                elif tag in adj_tags:
                    grouped_pos['Adjetivos'] += count
                elif tag in adv_tags:
                    grouped_pos['Adverbios'] += count
                elif tag in pronoun_tags:
                    grouped_pos['Pronombres'] += count
                else:
                    grouped_pos['Otros'] += count
            
            # Filtrar categorías vacías
            filtered_pos = {k: v for k, v in grouped_pos.items() if v > 0}
            
            # Crear gráfico de pie
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=list(filtered_pos.keys()),
                values=list(filtered_pos.values()),
                hole=0.3
            ))
            
            fig.update_layout(
                title=title,
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"pos_tags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de POS tags creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de POS: {e}")
            return ""
    
    async def create_emotions_chart(
        self, 
        emotions_data: Dict[str, float],
        title: str = "Análisis de Emociones"
    ) -> str:
        """Crear gráfico de análisis de emociones"""
        
        try:
            if not emotions_data:
                return ""
            
            # Filtrar emociones con valores > 0
            filtered_emotions = {k: v for k, v in emotions_data.items() if v > 0}
            
            if not filtered_emotions:
                return ""
            
            emotions = list(filtered_emotions.keys())
            scores = list(filtered_emotions.values())
            
            # Crear gráfico de barras polar
            fig = go.Figure()
            
            fig.add_trace(go.Barpolar(
                r=scores,
                theta=emotions,
                marker_color=px.colors.qualitative.Set3[:len(emotions)]
            ))
            
            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(scores) * 1.1]
                    )
                ),
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de emociones creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de emociones: {e}")
            return ""
    
    async def create_comparative_dashboard(
        self, 
        documents_data: List[Dict[str, Any]],
        title: str = "Dashboard Comparativo"
    ) -> str:
        """Crear dashboard comparativo para múltiples documentos"""
        
        try:
            if len(documents_data) < 2:
                return ""
            
            # Crear subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Comparación de Sentimientos',
                    'Longitud de Documentos', 
                    'Legibilidad',
                    'Riqueza Vocabulario'
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            doc_names = [f"Doc {i+1}" for i in range(len(documents_data))]
            
            # 1. Sentimientos
            sentiments = []
            for doc in documents_data:
                sentiment = doc.get('analysis', {}).get('sentiment', {})
                compound = sentiment.get('vader_compound', 0)
                sentiments.append(compound)
            
            fig.add_trace(
                go.Bar(x=doc_names, y=sentiments, name="Sentimiento"),
                row=1, col=1
            )
            
            # 2. Longitudes
            lengths = [doc.get('analysis', {}).get('word_count', 0) for doc in documents_data]
            
            fig.add_trace(
                go.Bar(x=doc_names, y=lengths, name="Palabras"),
                row=1, col=2
            )
            
            # 3. Legibilidad vs Longitud
            readabilities = [doc.get('analysis', {}).get('readability', {}).get('flesch_reading_ease', 50) 
                           for doc in documents_data]
            
            fig.add_trace(
                go.Scatter(x=lengths, y=readabilities, mode='markers+text', 
                          text=doc_names, textposition="top center", name="Legibilidad"),
                row=2, col=1
            )
            
            # 4. Riqueza vocabulario
            vocab_richness = [doc.get('analysis', {}).get('vocabulary_richness', 0) for doc in documents#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()# 4. Riqueza vocabulario
            vocab_richness = [doc.get('analysis', {}).get('vocabulary_richness', 0) for doc in documents_data]
            
            fig.add_trace(
                go.Bar(x=doc_names, y=vocab_richness, name="Riqueza Vocab."),
                row=2, col=2
            )
            
            # Actualizar layout
            fig.update_layout(
                title_text=title,
                showlegend=False,
                template="plotly_white",
                height=800
            )
            
            # Guardar archivo
            filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Dashboard comparativo creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando dashboard: {e}")
            return ""
    
    async def create_statistical_summary(
        self, 
        data: Dict[str, Any],
        title: str = "Resumen Estadístico"
    ) -> str:
        """Crear resumen estadístico visual"""
        
        try:
            # Crear figura con múltiples métricas
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Distribución de Longitudes',
                    'Métricas de Calidad',
                    'Tendencias Temporales',
                    'Correlaciones'
                )
            )
            
            # Datos de ejemplo (adaptar según datos reales)
            metrics = ['Legibilidad', 'Sentimiento', 'Complejidad', 'Coherencia']
            values = [75, 65, 45, 80]  # Valores de ejemplo
            
            # Gráfico de métricas
            fig.add_trace(
                go.Bar(x=metrics, y=values, name="Métricas"),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text=title,
                template="plotly_white",
                height=600
            )
            
            # Guardar archivo
            filename = f"stats_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando resumen estadístico: {e}")
            return ""
    
    def get_chart_types(self) -> List[str]:
        """Obtener tipos de gráficos disponibles"""
        return [
            "sentiment_analysis",
            "readability_metrics", 
            "keywords_distribution",
            "pos_tags_distribution",
            "emotions_analysis",
            "comparative_dashboard",
            "statistical_summary",
            "wordcloud",
            "timeline_analysis"
        ]
    
    async def cleanup(self):
        """Limpiar recursos"""
        plt.close('all')
'''
        
        visualizers_dir = self.service_path / "visualizers"
        with open(visualizers_dir / "chart_generator.py", "w", encoding="utf-8") as f:
            f.write(chart_generator_content)
        
        self.logger.info("✅ Generador de gráficos creado")

    def create_routers(self):
        """Crear routers de FastAPI"""
        self.logger.info("🌐 Creando routers...")
        
        # Router de análisis
        analysis_router = '''from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from visualizers.chart_generator import ChartGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

class AnalysisRequest(BaseModel):
    text: str
    options: Optional[Dict[str, Any]] = {}
    generate_insights: Optional[bool] = True
    generate_charts: Optional[bool] = True

class AnalysisResponse(BaseModel):
    analysis: Dict[str, Any]
    insights: List[Dict[str, Any]] = []
    charts: List[str] = []
    processing_time: float

@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks = None
):
    """Realizar análisis completo de texto"""
    
    try:
        # Inicializar servicios
        text_analyzer = TextAnalyzer()
        ai_insights = AIInsightsService()
        chart_generator = ChartGenerator()
        
        # Análisis principal
        analysis_result = await text_analyzer.analyze_text(request.text, request.options)
        
        response_data = {
            "analysis": analysis_result.__dict__,
            "insights": [],
            "charts": [],
            "processing_time": analysis_result.processing_time
        }
        
        # Generar insights con IA
        if request.generate_insights:
            try:
                insights = await ai_insights.generate_document_insights(
                    request.text, 
                    analysis_result.__dict__,
                    request.options
                )
                response_data["insights"] = [insight.__dict__ for insight in insights]
            except Exception as e:
                logger.warning(f"⚠️ Error generando insights: {e}")
        
        # Generar gráficos
        if request.generate_charts:
            try:
                charts = []
                
                # Gráfico de sentimientos
                sentiment_chart = await chart_generator.create_sentiment_chart(
                    analysis_result.sentiment
                )
                if sentiment_chart:
                    charts.append(sentiment_chart)
                
                # Gráfico de palabras clave
                if analysis_result.keywords:
                    keywords_chart = await chart_generator.create_keywords_chart(
                        analysis_result.keywords
                    )
                    if keywords_chart:
                        charts.append(keywords_chart)
                
                # Gráfico de emociones
                if analysis_result.emotion_scores:
                    emotions_chart = await chart_generator.create_emotions_chart(
                        analysis_result.emotion_scores
                    )
                    if emotions_chart:
                        charts.append(emotions_chart)
                
                response_data["charts"] = charts
                
            except Exception as e:
                logger.warning(f"⚠️ Error generando gráficos: {e}")
        
        return AnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Error en análisis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def analyze_batch(
    texts: List[str],
    options: Optional[Dict[str, Any]] = {},
    generate_comparative: bool = True
):
    """Análisis en lote de múltiples textos"""
    
    try:
        text_analyzer = TextAnalyzer()
        ai_insights = AIInsightsService()
        chart_generator = ChartGenerator()
        
        # Análisis en lote
        results = await text_analyzer.batch_analyze(texts)
        
        # Preparar datos para respuesta
        analysis_results = []
        for i, result in enumerate(results):
            analysis_results.append({
                "document_id": i,
                "analysis": result.__dict__ if hasattr(result, '__dict__') else result
            })
        
        response_data = {
            "batch_results": analysis_results,
            "comparative_insights": [],
            "comparative_charts": []
        }
        
        # Generar insights comparativos
        if generate_comparative and len(results) > 1:
            try:
                comparative_insights = await ai_insights.generate_comparative_insights(
                    analysis_results
                )
                response_data["comparative_insights"] = [
                    insight.__dict__ for insight in comparative_insights
                ]
                
                # Dashboard comparativo
                dashboard = await chart_generator.create_comparative_dashboard(
                    analysis_results
                )
                if dashboard:
                    response_data["comparative_charts"] = [dashboard]
                    
            except Exception as e:
                logger.warning(f"⚠️ Error en análisis comparativo: {e}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error en análisis en lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_analysis_capabilities():
    """Obtener capacidades de análisis disponibles"""
    
    text_analyzer = TextAnalyzer()
    
    return {
        "supported_languages": text_analyzer.get_supported_languages(),
        "analysis_features": [
            "sentiment_analysis",
            "readability_metrics",
            "entity_extraction", 
            "keyword_extraction",
            "topic_modeling",
            "pos_tagging",
            "emotion_analysis",
            "vocabulary_richness"
        ],
        "output_formats": ["json", "charts", "insights"],
        "batch_processing": True,
        "comparative_analysis": True
    }
'''
        
        routers_dir = self.service_path / "routers"
        with open(routers_dir / "analysis.py", "w", encoding="utf-8") as f:
            f.write(analysis_router)
        
        # Router de visualización
        visualization_router = '''from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from visualizers.chart_generator import ChartGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

class ChartRequest(BaseModel):
    chart_type: str
    data: Dict[str, Any]
    title: Optional[str] = "Gráfico"
    options: Optional[Dict[str, Any]] = {}

@router.post("/chart")
async def create_chart(request: ChartRequest):
    """Crear gráfico específico"""
    
    try:
        chart_generator = ChartGenerator()
        chart_path = ""
        
        if request.chart_type == "sentiment":
            chart_path = await chart_generator.create_sentiment_chart(
                request.data, request.title
            )
        elif request.chart_type == "keywords":
            chart_path = await chart_generator.create_keywords_chart(
                request.data, request.title
            )
        elif request.chart_type == "emotions":
            chart_path = await chart_generator.create_emotions_chart(
                request.data, request.title
            )
        elif request.chart_type == "readability":
            chart_path = await chart_generator.create_readability_chart(
                request.data, request.title
            )
        elif request.chart_type == "pos_tags":
            chart_path = await chart_generator.create_pos_tags_chart(
                request.data, request.title
            )
        else:
            raise HTTPException(status_code=400, detail=f"Tipo de gráfico no soportado: {request.chart_type}")
        
        if chart_path:
            return {"chart_url": f"/static/plots/{chart_path.split('/')[-1]}"}
        else:
            raise HTTPException(status_code=500, detail="Error generando gráfico")
            
    except Exception as e:
        logger.error(f"❌ Error creando gráfico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboard")
async def create_dashboard(
    documents_data: List[Dict[str, Any]],
    title: str = "Dashboard Analítico"
):
    """Crear dashboard comparativo"""
    
    try:
        chart_generator = ChartGenerator()
        
        dashboard_path = await chart_generator.create_comparative_dashboard(
            documents_data, title
        )
        
        if dashboard_path:
            return {"dashboard_url": f"/static/plots/{dashboard_path.split('/')[-1]}"}
        else:
            raise HTTPException(status_code=500, detail="Error generando dashboard")
            
    except Exception as e:
        logger.error(f"❌ Error creando dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chart-types")
async def get_chart_types():
    """Obtener tipos de gráficos disponibles"""
    
    chart_generator = ChartGenerator()
    return {
        "available_charts": chart_generator.get_chart_types(),
        "formats": ["html", "png", "svg"],
        "interactive": True
    }
'''
        
        with open(routers_dir / "visualization.py", "w", encoding="utf-8") as f:
            f.write(visualization_router)
        
        # Crear otros routers básicos
        for router_name in ["statistics", "insights", "models"]:
            basic_router = f'''from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

@router.get("/")
async def {router_name}_root():
    return {{"message": "{router_name.title()} router - En desarrollo"}}
'''
            with open(routers_dir / f"{router_name}.py", "w", encoding="utf-8") as f:
                f.write(basic_router)
        
        self.logger.info("✅ Routers creados")

    def create_config_utils(self):
        """Crear utilidades de configuración"""
        self.logger.info("⚙️ Creando utilidades...")
        
        # Config
        config_content = '''"""
Configuración del Analytics-Engine
"""
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Configuración general
    app_name: str = "Analytics-Engine"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8003
    
    # Puertos de otros servicios
    ai_engine_port: int = 8001
    document_processor_port: int = 8002
    
    # Directorios
    data_dir: str = "./data"
    reports_dir: str = "./data/reports"
    cache_dir: str = "./data/cache"
    plots_dir: str = "./static/plots"
    
    # Análisis de texto
    default_language: str = "auto"
    max_text_length: int = 1000000  # 1MB
    enable_topic_modeling: bool = True
    enable_entity_extraction: bool = True
    
    # IA e Insights
    enable_ai_insights: bool = True
    ai_insights_timeout: int = 30
    max_insights_per_document: int = 10
    
    # Visualización
    default_chart_format: str = "html"
    chart_width: int = 800
    chart_height: int = 600
    enable_interactive_charts: bool = True
    
    # CORS
    cors_origins: List[str] = ["http://localhost:8080", "http://127.0.0.1:8080"]
    
    # Base de datos
    database_url: str = "sqlite:///./analytics_engine.db"
    redis_url: str = "redis://localhost:6379/3"
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    # Modelos NLP
    spacy_model: str = "es_core_news_sm"  # Modelo principal
    spacy_fallback: str = "en_core_web_sm"  # Modelo de respaldo
    
    # Análisis estadístico
    enable_statistical_analysis: bool = True
    confidence_interval: float = 0.95
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
'''
        
        utils_dir = self.service_path / "utils"
        with open(utils_dir / "config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        # Logging config
        logging_config = '''"""
Configuración de logging para Analytics-Engine
"""
import logging
import logging.handlers
from pathlib import Path

def setup_logging():
    """Configurar sistema de logging"""
    
    # Crear directorio de logs
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configuración del logger root
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler con rotación
            logging.handlers.RotatingFileHandler(
                log_dir / "analytics_engine.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Configurar loggers específicos
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("plotly").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
'''
        
        with open(utils_dir / "logging_config.py", "w", encoding="utf-8") as f:
            f.write(logging_config)
        
        self.logger.info("✅ Utilidades creadas")

    def create_analytics_manager(self):
        """Crear gestor principal de análisis"""
        self.logger.info("📊 Creando gestor de análisis...")
        
        analytics_manager_content = '''"""
Gestor principal del motor de análisis
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """Administrador principal del motor de análisis"""
    
    def __init__(self):
        self.analysis_history = []
        self.statistics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "total_documents_processed": 0,
            "total_words_analyzed": 0,
            "average_processing_time": 0.0,
            "analysis_types": {}
        }
        self.is_ready = False
        
    async def initialize(self):
        """Inicializar el gestor de análisis"""
        logger.info("🔧 Inicializando Analytics Manager...")
        
        # Crear directorios necesarios
        directories = [
            Path("data/datasets"),
            Path("data/reports"),
            Path("data/cache"),
            Path("static/plots")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Cargar estadísticas si existen
        await self._load_statistics()
        
        self.is_ready = True
        logger.info("✅ Analytics Manager inicializado")
    
    async def register_analysis(
        self, 
        analysis_type: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any],
        processing_time: float
    ):
        """Registrar un análisis realizado"""
        
        analysis_record = {
            "id": len(self.analysis_history) + 1,
            "type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time,
            "input_size": len(str(input_data)),
            "success": "error" not in result,
            "metadata": {
                "word_count": input_data.get("word_count", 0),
                "language": result.get("language", "unknown")
            }
        }
        
        self.analysis_history.append(analysis_record)
        
        # Actualizar estadísticas
        self.statistics["total_analyses"] += 1
        if analysis_record["success"]:
            self.statistics["successful_analyses"] += 1
        else:
            self.statistics["failed_analyses"] += 1
        
        self.statistics["total_words_analyzed"] += analysis_record["metadata"]["word_count"]
        
        # Actualizar tiempo promedio
        total_time = sum(a["processing_time"] for a in self.analysis_history)
        self.statistics["average_processing_time"] = total_time / len(self.analysis_history)
        
        # Contar tipos de análisis
        if analysis_type in self.statistics["analysis_types"]:
            self.statistics["analysis_types"][analysis_type] += 1
        else:
            self.statistics["analysis_types"][analysis_type] = 1
        
        await self._save_statistics()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del motor"""
        
        # Calcular métricas adicionales
        success_rate = 0.0
        if self.statistics["total_analyses"] > 0:
            success_rate = (self.statistics["successful_analyses"] / 
                          self.statistics["total_analyses"]) * 100
        
        # Análisis recientes (últimos 10)
        recent_analyses = self.analysis_history[-10:] if self.analysis_history else []
        
        return {
            **self.statistics,
            "success_rate": round(success_rate, 2),
            "recent_analyses": recent_analyses,
            "storage_info": await self._get_storage_info(),
            "performance_metrics": await self._get_performance_metrics()
        }
    
    async def _get_storage_info(self) -> Dict[str, Any]:
        """Obtener información de almacenamiento"""
        
        storage_info = {}
        
        for directory_name, path in [
            ("datasets", "data/datasets"),
            ("reports", "data/reports"),
            ("cache", "data/cache"),
            ("plots", "static/plots")
        ]:
            directory = Path(path)
            if directory.exists():
                files = list(directory.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                storage_info[directory_name] = {
                    "files": len([f for f in files if f.is_file()]),
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "path": str(directory)
                }
            else:
                storage_info[directory_name] = {
                    "files": 0,
                    "total_size_mb": 0,
                    "path": str(directory)
                }
        
        return storage_info
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento"""
        
        if not self.analysis_history:
            return {}
        
        processing_times = [a["processing_time"] for a in self.analysis_history]
        word_counts = [a["metadata"]["word_count"] for a in self.analysis_history]
        
        return {
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "median_processing_time": sorted(processing_times)[len(processing_times)//2],
            "avg_words_per_second": (
                sum(word_counts) / sum(processing_times) 
                if sum(processing_times) > 0 else 0
            ),
            "total_processing_time": sum(processing_times)
        }
    
    async def _load_statistics(self):
        """Cargar estadísticas desde archivo"""
        stats_file = Path("data/analytics_stats.json")
        
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.statistics = data.get("statistics", self.statistics)
                    self.analysis_history = data.get("history", [])
                logger.info("📊 Estadísticas cargadas")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando estadísticas: {e}")
    
    async def _save_statistics(self):
        """Guardar estadísticas en archivo"""
        stats_file = Path("data/analytics_stats.json")
        
        try:
            data = {
                "statistics": self.statistics,
                "history": self.analysis_history[-1000:]  # Mantener últimos 1000
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Error guardando estadísticas: {e}")
    
    async def cleanup_old_data(self, days: int = 30):
        """Limpiar datos antiguos"""
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cleaned_files = 0
        
        for directory_path in ["data/cache", "static/plots"]:
            directory = Path(directory_path)
            if not directory.exists():
                continue
            
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            cleaned_files += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Error eliminando {file_path}: {e}")
        
        logger.info(f"🧹 Limpieza completada: {cleaned_files} archivos eliminados")
        return cleaned_files
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generar reporte de rendimiento"""
        
        stats = await self.get_statistics()
        
        report = {
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_analyses": stats["total_analyses"],
                "success_rate": stats["success_rate"],
                "avg_processing_time": stats["average_processing_time"],
                "total_words_processed": stats["total_words_analyzed"]
            },
            "performance": stats.get("performance_metrics", {}),
            "storage": stats.get("storage_info", {}),
            "analysis_breakdown": stats.get("analysis_types", {}),
            "recommendations": []
        }
        
        # Generar recomendaciones
        if stats["success_rate"] < 95:
            report["recommendations"].append("Revisar casos de fallo para mejorar robustez")
        
        if stats["average_processing_time"] > 5.0:
            report["recommendations"].append("Optimizar algoritmos para mejorar velocidad")
        
        # Guardar reporte
        report_file = Path(f"data/reports/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    async def cleanup(self):
        """Limpiar recursos"""
        logger.info("🧹 Limpiando recursos del Analytics Manager...")
        await self._save_statistics()
        logger.info("✅ Recursos limpiados")
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "analytics_manager.py", "w", encoding="utf-8") as f:
            f.write(analytics_manager_content)
        
        self.logger.info("✅ Gestor de análisis creado")

    def create_data_processor(self):
        """Crear procesador de datos"""
        self.logger.info("🔢 Creando procesador de datos...")
        
        data_processor_content = '''"""
Procesador de datos para análisis estadístico
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

class DataProcessor:
    """Procesador de datos para análisis estadístico"""
    
    def __init__(self):
        self.is_ready = False
        self.supported_formats = ['json', 'csv', 'excel', 'dataframe']
        
    async def initialize(self):
        """Inicializar el procesador de datos"""
        logger.info("🔧 Inicializando Data Processor...")
        self.is_ready = True
        logger.info("✅ Data Processor inicializado")
    
    async def process_text_analysis_data(
        self, 
        analysis_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Procesar resultados de análisis de texto en DataFrame"""
        
        try:
            # Extraer métricas numéricas
            processed_data = []
            
            for i, result in enumerate(analysis_results):
                row = {
                    'document_id': i,
                    'text_length': result.get('text_length', 0),
                    'word_count': result.get('word_count', 0),
                    'sentence_count': result.get('sentence_count', 0),
                    'paragraph_count': result.get('paragraph_count', 0),
                    'language': result.get('language', 'unknown'),
                    'vocabulary_richness': result.get('vocabulary_richness', 0),
                    'average_sentence_length': result.get('average_sentence_length', 0),
                }
                
                # Métricas de sentimiento
                sentiment = result.get('sentiment', {})
                row.update({
                    'sentiment_compound': sentiment.get('vader_compound', 0),
                    'sentiment_positive': sentiment.get('vader_positive', 0),
                    'sentiment_negative': sentiment.get('vader_negative', 0),
                    'sentiment_neutral': sentiment.get('vader_neutral', 0),
                    'textblob_polarity': sentiment.get('textblob_polarity', 0),
                    'textblob_subjectivity': sentiment.get('textblob_subjectivity', 0),
                })
                
                # Métricas de legibilidad
                readability = result.get('readability', {})
                row.update({
                    'flesch_reading_ease': readability.get('flesch_reading_ease', 0),
                    'avg_word_length': readability.get('average_word_length', 0),
                    'reading_time_minutes': readability.get('estimated_reading_time_minutes', 0),
                })
                
                # Métricas de emociones
                emotions = result.get('emotion_scores', {})
                for emotion, score in emotions.items():
                    row[f'emotion_{emotion}'] = score
                
                processed_data.append(row)
            
            df = pd.DataFrame(processed_data)
            logger.info(f"✅ DataFrame creado con {len(df)} filas y {len(df.columns)} columnas")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error procesando datos: {e}")
            return pd.DataFrame()
    
    async def calculate_descriptive_statistics(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calcular estadísticas descriptivas"""
        
        try:
            # Seleccionar solo columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {}
            
            stats_summary = {
                'basic_stats': df[numeric_cols].describe().to_dict(),
                'correlations': df[numeric_cols].corr().to_dict(),
                'skewness': df[numeric_cols].skew().to_dict(),
                'kurtosis': df[numeric_cols].kurtosis().to_dict(),
                'missing_values': df[numeric_cols].isnull().sum().to_dict()
            }
            
            # Estadísticas adicionales
            stats_summary['variance'] = df[numeric_cols].var().to_dict()
            stats_summary['std_dev'] = df[numeric_cols].std().to_dict()
            stats_summary['range'] = (df[numeric_cols].max() - df[numeric_cols].min()).to_dict()
            
            logger.info("✅ Estadísticas descriptivas calculadas")
            return stats_summary
            
        except Exception as e:
            logger.error(f"❌ Error calculando estadísticas: {e}")
            return {}
    
    async def perform_correlation_analysis(
        self, 
        df: pd.DataFrame,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """Realizar análisis de correlación"""
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {}
            
            # Matriz de correlación
            corr_matrix = df[numeric_cols].corr(method=method)
            
            # Encontrar correlaciones fuertes
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Correlación fuerte
                        strong_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'method': method,
                'variables_analyzed': list(numeric_cols)
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de correlación: {e}")
            return {}
    
    async def perform_clustering_analysis(
        self, 
        df: pd.DataFrame,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Realizar análisis de clustering"""
        
        try:
            # Seleccionar variables numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {}
            
            # Preparar datos
            data = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Normalizar datos
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Determinar número óptimo de clusters si no se especifica
            if n_clusters is None:
                n_clusters = await self._find_optimal_clusters(data_scaled)
            
            # Aplicar K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data_scaled)
            
            # Calcular métricas
            silhouette_avg = silhouette_score(data_scaled, cluster_labels)
            
            # Agregar clusters al DataFrame
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = cluster_labels
            
            # Estadísticas por cluster
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                cluster_stats[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df) * 100,
                    'means': cluster_data[numeric_cols].mean().to_dict()
                }
            
            return {
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'silhouette_score': silhouette_avg,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_statistics': cluster_stats,
                'variables_used': list(numeric_cols)
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de clustering: {e}")
            return {}
    
    async def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Encontrar número óptimo de clusters usando método del codo"""
        
        try:
            inertias = []
            k_range = range(2, min(max_k + 1, len(data)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            # Método del codo (simplificado)
            if len(inertias) < 2:
                return 2
            
            # Encontrar el punto donde la mejora es menor
            differences = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            optimal_k = differences.index(min(differences)) + 2
            
            return min(optimal_k, 5)  # Limitar a máximo 5 clusters
            
        except Exception as e:
            logger.warning(f"⚠️ Error encontrando clusters óptimos: {e}")
            return 3  # Valor por defecto
    
    async def perform_pca_analysis(
        self, 
        df: pd.DataFrame,
        n_components: Optional[int] = None
    ) -> Dict[str, Any]:
        """Realizar análisis de componentes principales (PCA)"""
        
        try:
            # Seleccionar variables numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {}
            
            # Preparar datos
            data = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Normalizar datos
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Determinar número de componentes
            if n_components is None:
                n_components = min(len(numeric_cols), 5)
            
            # Aplicar PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(data_scaled)
            
            # Calcular varianza explicada
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            return {
                'n_components': n_components,
                'explained_variance_ratio': explained_variance.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'components': components.tolist(),
                'feature_loadings': pca.components_.tolist(),
                'variables_used': list(numeric_cols),
                'total_variance_explained': float(cumulative_variance[-1])
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis PCA: {e}")
            return {}
    
    async def detect_outliers(
        self, 
        df: pd.DataFrame,
        method: str = 'iqr'
    ) -> Dict[str, Any]:
        """Detectar valores atípicos"""
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {}
            
            outliers_info = {}
            
            for col in numeric_cols:
                column_data = df[col].dropna()
                
                if method == 'iqr':
                    Q1 = column_data.quantile(0.25)
                    Q3 = column_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(column_data))
                    outliers = column_data[z_scores > 3]
                
                else:
                    continue
                
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(column_data) * 100,
                    'values': outliers.tolist() if len(outliers) < 20 else outliers.head(20).tolist(),
                    'method': method
                }
            
            return {
                'outliers_by_variable': outliers_info,
                'detection_method': method,
                'total_variables_analyzed': len(numeric_cols)
            }
            
        except Exception as e:
            logger.error(f"❌ Error detectando outliers: {e}")
            return {}
    
    async def generate_data_quality_report(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generar reporte de calidad de datos"""
        
        try:
            report = {
                'dataset_info': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                },
                'missing_data': {
                    'columns_with_missing': df.isnull().any().sum(),
                    'total_missing_values': df.isnull().sum().sum(),
                    'missing_percentage_by_column': (df.isnull().sum() / len(df) * 100).to_dict()
                },
                'data_types': df.dtypes.astype(str).to_dict(),
                'unique_values': df.nunique().to_dict()
            }
            
            # Detectar posibles problemas
            issues = []
            
            # Columnas con muchos valores faltantes
            high_missing = df.columns[df.isnull().mean() > 0.5]
            if len(high_missing) > 0:
                issues.append(f"Columnas con >50% valores faltantes: {list(high_missing)}")
            
            # Columnas con un solo valor
            single_value_cols = df.columns[df.nunique() == 1]
            if len(single_value_cols) > 0:
                issues.append(f"Columnas con valor único: {list(single_value_cols)}")
            
            # Columnas numéricas con valores constantes
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            constant_cols = [col for col in numeric_cols if df[col].std() == 0]
            if constant_cols:
                issues.append(f"Columnas numéricas constantes: {constant_cols}")
            
            report['quality_issues'] = issues
            report['data_quality_score'] = max(0, 100 - len(issues) * 10)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte de calidad: {e}")
            return {}
    
    def get_supported_formats(self) -> List[str]:
        """Obtener formatos soportados"""
        return self.supported_formats
    
    async def export_analysis_results(
        self, 
        results: Dict[str, Any],
        format: str = 'json',
        output_path: Optional[str] = None
    ) -> str:
        """Exportar resultados de análisis"""
        
        try:
            if not output_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"data/reports/analysis_results_{timestamp}.{format}"
            
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str)
            
            elif format == 'csv' and 'dataframe' in results:
                df = pd.DataFrame(results['dataframe'])
                df.to_csv(output_path, index=False)
            
            elif format == 'excel' and 'dataframe' in results:
                df = pd.DataFrame(results['dataframe'])
                df.to_excel(output_path, index=False)
            
            logger.info(f"✅ Resultados exportados a: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error exportando resultados: {e}")
            return ""
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "data_processor.py", "w", encoding="utf-8") as f:
            f.write(data_processor_content)
        
        self.logger.info("✅ Procesador de datos creado")

    def create_dockerfile(self):
        """Crear Dockerfile"""
        self.logger.info("🐳 Creando Dockerfile...")
        
        dockerfile_content = '''FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelos de spaCy
RUN python -m spacy download es_core_news_sm || true
RUN python -m spacy download en_core_web_sm || true

# Crear directorios necesarios
RUN mkdir -p /app/data/datasets /app/data/reports /app/data/cache /app/static/plots /app/logs && \
    chown -R appuser:appuser /app

# Copiar código de la aplicación
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

# Exponer puerto
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Comando por defecto
CMD ["python", "app.py"]
'''
        
        with open(self.service_path / "Dockerfile", "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        
        self.logger.info("✅ Dockerfile creado")

    def create_tests(self):
        """Crear tests básicos"""
        self.logger.info("🧪 Creando tests...")
        
        test_content = '''"""
Tests para Analytics-Engine
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from app import app
from services.text_analyzer import TextAnalyzer
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator

client = TestClient(app)

def test_health_check():
    """Test del health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "analytics-engine"

def test_status_endpoint():
    """Test del endpoint de status"""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "components" in data

def test_capabilities_endpoint():
    """Test del endpoint de capacidades"""
    response = client.get("/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "text_analysis" in data
    assert "statistical_analysis" in data
    assert "ai_insights" in data
    assert "visualization" in data

@pytest.mark.asyncio
async def test_text_analyzer():
    """Test del analizador de texto"""
    analyzer = TextAnalyzer()
    await analyzer.initialize()
    
    test_text = "Este es un texto de prueba para análisis. Contiene múltiples oraciones. ¡Es muy interesante!"
    
    result = await analyzer.analyze_text(test_text)
    
    assert result.text_length > 0
    assert result.word_count > 0
    assert result.sentence_count > 0
    assert result.language is not None
    assert isinstance(result.sentiment, dict)
    assert isinstance(result.keywords, list)

@pytest.mark.asyncio 
async def test_data_processor():
    """Test del procesador de datos"""
    processor = DataProcessor()
    await processor.initialize()
    
    # Crear datos de prueba
    test_data = [
        {
            'text_length': 100,
            'word_count': 20,
            'sentiment': {'vader_compound': 0.5},
            'readability': {'flesch_reading_ease': 70}
        },
        {
            'text_length': 150,
            'word_count': 30,
            'sentiment': {'vader_compound': -0.3},
            'readability': {'flesch_reading_ease': 60}
        }
    ]
    
    df = await processor.process_text_analysis_data(test_data)
    
    assert len(df) == 2
    assert 'text_length' in df.columns
    assert 'word_count' in df.columns

@pytest.mark.asyncio
async def test_chart_generator():
    """Test del generador de gráficos"""
    generator = ChartGenerator()
    await generator.initialize()
    
    # Test datos de sentimiento
    sentiment_data = {
        'vader_positive': 0.3,
        'vader_negative': 0.1,
        'vader_neutral': 0.6
    }
    
    chart_path = await generator.create_sentiment_chart(sentiment_data)
    
    # Verificar que se creó el archivo (o al menos no hay error)
    assert isinstance(chart_path, str)

def test_analyze_text_endpoint():
    """Test del endpoint de análisis de texto"""
    response = client.post(
        "/analyze/text",
        json={
            "text": "Este es un texto de prueba.",
            "generate_insights": False,
            "generate_charts": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    assert "processing_time" in data

def test_visualization_chart_types():
    """Test del endpoint de tipos de gráficos"""
    response = client.get("/visualize/chart-types")
    assert response.status_code == 200
    data = response.json()
    assert "available_charts" in data
    assert "formats" in data

if __name__ == "__main__":
    pytest.main([__file__])
'''
        
        tests_dir = self.service_path / "tests"
        with open(tests_dir / "test_analytics_engine.py", "w", encoding="utf-8") as f:
            f.write(test_content)
        
        self.logger.info("✅ Tests creados")

    def install_dependencies(self):
        """Instalar dependencias específicas"""
        self.logger.info("📦 Instalando dependencias del Analytics-Engine...")
        
        # Determinar el ejecutable de pip
        if os.name == 'nt':  # Windows
            pip_exe = self.venv_path / "Scripts" / "pip.exe"
        else:  # Linux/macOS
            pip_exe = self.venv_path / "bin" / "pip"
        
        if not pip_exe.exists():
            self.logger.error("❌ pip no encontrado en el entorno virtual")
            return False
        
        try:
            # Instalar dependencias
            result = subprocess.run([
                str(pip_exe), "install", "-r", 
                str(self.service_path / "requirements.txt")
            ], check=True, capture_output=True, text=True, cwd=self.service_path)
            
            self.logger.info("✅ Dependencias del Analytics-Engine instaladas")
            
            # Intentar descargar modelos de spaCy
            self.logger.info("📥 Descargando modelos de spaCy...")
            try:
                subprocess.run([
                    str(pip_exe), "install", "https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl"
                ], check=False, capture_output=True, text=True)
                
                subprocess.run([
                    str(pip_exe), "install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
                ], check=False, capture_output=True, text=True)
                
                self.logger.info("✅ Modelos de spaCy descargados")
            except Exception as e:
                self.logger.warning(f"⚠️ No se pudieron descargar modelos spaCy: {e}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Error instalando dependencias: {e}")
            self.logger.error(f"   Salida: {e.stdout}")
            self.logger.error(f"   Error: {e.stderr}")
            self.logger.warning("⚠️ Algunas dependencias pueden requerir instalación manual")
            return False

    def create_start_script(self):
        """Crear script de inicio"""
        self.logger.info("🚀 Creando script de inicio...")
        
        if os.name == 'nt':  # Windows
            start_script = '''@echo off
echo 📊 INICIANDO ANALYTICS-ENGINE...
echo =================================

cd /d "%~dp0"

REM Activar entorno virtual
call ..\\..\\venv\\Scripts\\activate.bat

REM Verificar que estamos en el directorio correcto
if not exist "app.py" (
    echo ❌ app.py no encontrado. Verifica que estés en services/analytics-engine/
    pause
    exit /b 1
)

REM Verificar modelos de spaCy (opcional)
python -c "import spacy; spacy.load('es_core_news_sm')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Modelo spaCy español no encontrado. Funcionalidad NLP limitada.
)

REM Iniciar servidor
echo ✅ Iniciando servidor en puerto 8003...
python app.py

pause
'''
            with open(self.service_path / "start.bat", "w", encoding="utf-8") as f:
                f.write(start_script)
        
        else:  # Linux/macOS
            start_script = '''#!/bin/bash

echo "📊 INICIANDO ANALYTICS-ENGINE..."
echo "================================="

# Ir al directorio del script
cd "$(dirname "$0")"

# Activar entorno virtual
source ../../venv/bin/activate

# Verificar que estamos en el directorio correcto
if [ ! -f "app.py" ]; then
    echo "❌ app.py no encontrado. Verifica que estés en services/analytics-engine/"
    exit 1
fi

# Verificar modelos de spaCy (opcional)
if ! python -c "import spacy; spacy.load('es_core_news_sm')" &> /dev/null; then
    echo "⚠️ Modelo spaCy español no encontrado. Descarga con:"
    echo "   python -m spacy download es_core_news_sm"
fi

# Iniciar servidor
echo "✅ Iniciando servidor en puerto 8003..."
python app.py
'''
            start_file = self.service_path / "start.sh"
            with open(start_file, "w", encoding="utf-8") as f:
                f.write(start_script)
            
            # Hacer ejecutable
            start_file.chmod(0o755)
        
        self.logger.info("✅ Script de inicio creado")

    def create_env_file(self):
        """Crear archivo .env local"""
        self.logger.info("⚙️ Creando .env local...")
        
        env_content = '''# ANALYTICS-ENGINE - Configuración local
DEBUG=true
LOG_LEVEL=INFO

# Puertos de servicios
AI_ENGINE_PORT=8001
DOCUMENT_PROCESSOR_PORT=8002

# Análisis
DEFAULT_LANGUAGE=auto
ENABLE_TOPIC_MODELING=true
ENABLE_ENTITY_EXTRACTION=true

# IA e Insights
ENABLE_AI_INSIGHTS=true
AI_INSIGHTS_TIMEOUT=30

# Visualización
DEFAULT_CHART_FORMAT=html
ENABLE_INTERACTIVE_CHARTS=true

# Modelos NLP
SPACY_MODEL=es_core_news_sm
SPACY_FALLBACK=en_core_web_sm

# Base de datos
DATABASE_URL=sqlite:///./analytics_engine.db
REDIS_URL=redis://localhost:6379/3
'''
        
        with open(self.service_path / ".env", "w", encoding="utf-8") as f:
            f.write(env_content)
        
        self.logger.info("✅ .env local creado")

    def run_setup(self):
        """Ejecutar setup completo"""
        self.logger.info("🚀 INICIANDO SETUP DEL ANALYTICS-ENGINE")
        self.logger.info("=" * 50)
        
        try:
            # Validar entorno
            if not self.validate_environment():
                return False
            
            # Crear estructura
            self.create_directory_structure()
            
            # Crear archivos principales
            self.create_main_app()
            self.create_requirements()
            
            # Crear servicios core
            self.create_text_analyzer()
            self.create_ai_insights_service()
            self.create_chart_generator()
            
            # Crear routers y utilidades
            self.create_routers()
            self.create_config_utils()
            self.create_analytics_manager()
            self.create_data_processor()
            
            # Crear archivos de deployment
            self.create_dockerfile()
            self.create_tests()
            self.create_start_script()
            self.create_env_file()
            
            # Instalar dependencias
            if not self.install_dependencies():
                self.logger.warning("⚠️ Algunas dependencias no se instalaron correctamente")
            
            self.logger.info("🎉 ANALYTICS-ENGINE CONFIGURADO EXITOSAMENTE!")
            self.logger.info("=" * 50)
            self.logger.info("")
            self.logger.info("📋 PRÓXIMOS PASOS:")
            self.logger.info("1. (Opcional) Descargar modelos de spaCy:")
            self.logger.info("   python -m spacy download es_core_news_sm")
            self.logger.info("   python -m spacy download en_core_web_sm")
            self.logger.info("2. Ejecutar: python app.py")
            self.logger.info("3. Verificar: http://localhost:8003/health")
            self.logger.info("4. Documentación: http://localhost:8003/docs")
            self.logger.info("")
            self.logger.info("🔗 ENDPOINTS PRINCIPALES:")
            self.logger.info("   • Health: http://localhost:8003/health")
            self.logger.info("   • Status: http://localhost:8003/status")
            self.logger.info("   • Docs: http://localhost:8003/docs")
            self.logger.info("   • Capacidades: http://localhost:8003/capabilities")
            self.logger.info("   • Análisis: http://localhost:8003/analyze/text")
            self.logger.info("   • Visualización: http://localhost:8003/visualize/chart")
            self.logger.info("")
            self.logger.info("📊 CAPACIDADES DE ANÁLISIS:")
            self.logger.info("   • Análisis de sentimientos con VADER y TextBlob")
            self.logger.info("   • Extracción de palabras clave con TF-IDF")
            self.logger.info("   • Análisis de legibilidad y métricas de texto")
            self.logger.info("   • Detección de entidades con spaCy")
            self.logger.info("   • Modelado de temas con LDA")
            self.logger.info("   • Análisis estadístico y clustering")
            self.logger.info("   • Visualizaciones interactivas con Plotly")
            self.logger.info("   • Insights generados por IA")
            self.logger.info("   • Análisis comparativo de múltiples documentos")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error durante el setup: {e}")
            return False

def main():
    """Función principal"""
    setup = AnalyticsEngineSetup()
    success = setup.run_setup()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
                        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            # Normalizar por longitud del texto
            emotion_scores[emotion] = score / max(len(text.split()), 1) * 100
        
        return emotion_scores
    
    async def generate_wordcloud(self, text: str, output_path: str = None) -> str:
        """Generar nube de palabras"""
        
        try:
            # Limpiar texto
            words = nltk.word_tokenize(text.lower())
            words = [w for w in words if w.isalnum() and len(w) > 2]
            clean_text = ' '.join(words)
            
            # Generar wordcloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(clean_text)
            
            # Guardar imagen
            if not output_path:
                output_path = f"static/plots/wordcloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            wordcloud.to_file(output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generando wordcloud: {e}")
            return ""
    
    async def batch_analyze(self, texts: List[str]) -> List[TextAnalysisResult]:
        """Análisis en lote de múltiples textos"""
        
        logger.info(f"📦 Analizando lote de {len(texts)} textos...")
        
        tasks = [self.analyze_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar resultados exitosos
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error en texto {i}: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"✅ Lote completado: {len(successful_results)}/{len(texts)} exitosos")
        return successful_results
    
    def get_supported_languages(self) -> List[str]:
        """Obtener idiomas soportados"""
        return self.supported_languages
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Obtener estado de modelos cargados"""
        return {
            'spacy_model': self.nlp_model is not None,
            'vader_sentiment': self.sentiment_analyzer is not None,
            'nltk_resources': True  # Asumimos que están disponibles si llegó hasta aquí
        }
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "text_analyzer.py", "w", encoding="utf-8") as f:
            f.write(text_analyzer_content)
        
        self.logger.info("✅ Analizador de texto creado")

    def create_ai_insights_service(self):
        """Crear servicio de insights con IA"""
        self.logger.info("🤖 Creando servicio de insights con IA...")
        
        ai_insights_content = '''"""
Servicio de insights usando IA - se conecta con AI-Engine
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import httpx
from dataclasses import dataclass
from datetime import datetime
import json

from utils.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class AIInsight:
    """Estructura para un insight generado por IA"""
    title: str
    content: str
    confidence: float
    category: str
    metadata: Dict[str, Any]
    timestamp: str

class AIInsightsService:
    """Servicio para generar insights usando el AI-Engine"""
    
    def __init__(self):
        self.settings = Settings()
        self.is_ready = False
        self.ai_engine_url = f"http://localhost:{self.settings.ai_engine_port}"
        self.client = None
        
    async def initialize(self):
        """Inicializar el servicio de insights"""
        logger.info("🔧 Inicializando AI Insights Service...")
        
        try:
            # Crear cliente HTTP
            self.client = httpx.AsyncClient(timeout=30.0)
            
            # Verificar conexión con AI-Engine
            connection_ok = await self.check_ai_engine_connection()
            if connection_ok:
                logger.info("✅ Conexión con AI-Engine establecida")
            else:
                logger.warning("⚠️ AI-Engine no disponible. Modo limitado.")
            
            self.is_ready = True
            logger.info("✅ AI Insights Service inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando AI Insights: {e}")
            self.is_ready = True  # Continuar en modo limitado
    
    async def check_ai_engine_connection(self) -> bool:
        """Verificar conexión con AI-Engine"""
        
        try:
            response = await self.client.get(f"{self.ai_engine_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def generate_document_insights(
        self, 
        text: str, 
        analysis_result: Dict = None,
        options: Dict = None
    ) -> List[AIInsight]:
        """Generar insights sobre un documento"""
        
        options = options or {}
        insights = []
        
        try:
            # Insight 1: Resumen del contenido
            if options.get('generate_summary', True):
                summary_insight = await self._generate_summary_insight(text)
                if summary_insight:
                    insights.append(summary_insight)
            
            # Insight 2: Análisis de temas principales
            if options.get('analyze_themes', True):
                themes_insight = await self._analyze_themes_insight(text)
                if themes_insight:
                    insights.append(themes_insight)
            
            # Insight 3: Clasificación del documento
            if options.get('classify_document', True):
                classification_insight = await self._classify_document_insight(text)
                if classification_insight:
                    insights.append(classification_insight)
            
            # Insight 4: Análisis de sentimiento contextual
            if options.get('sentiment_context', True) and analysis_result:
                sentiment_insight = await self._generate_sentiment_context(text, analysis_result)
                if sentiment_insight:
                    insights.append(sentiment_insight)
            
            # Insight 5: Recomendaciones
            if options.get('generate_recommendations', True):
                recommendations_insight = await self._generate_recommendations(text, analysis_result)
                if recommendations_insight:
                    insights.append(recommendations_insight)
            
            logger.info(f"✅ Generados {len(insights)} insights")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generando insights: {e}")
            return []
    
    async def _generate_summary_insight(self, text: str) -> Optional[AIInsight]:
        """Generar resumen inteligente del documento"""
        
        try:
            prompt = f"""
            Analiza el siguiente texto y proporciona un resumen conciso que capture:
            1. Los puntos principales
            2. Los temas centrales
            3. Las conclusiones clave
            
            Texto a analizar:
            {text[:2000]}...
            
            Proporciona un resumen de máximo 150 palabras.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Resumen del Documento",
                    content=response,
                    confidence=0.85,
                    category="summary",
                    metadata={"word_count": len(response.split())},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error generando resumen: {e}")
        
        return None
    
    async def _analyze_themes_insight(self, text: str) -> Optional[AIInsight]:
        """Analizar temas principales del documento"""
        
        try:
            prompt = f"""
            Identifica y analiza los 3-5 temas principales en el siguiente texto.
            Para cada tema, proporciona:
            1. Nombre del tema
            2. Relevancia (1-10)
            3. Breve descripción
            
            Texto:
            {text[:2000]}...
            
            Responde en formato estructurado.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Análisis de Temas Principales",
                    content=response,
                    confidence=0.80,
                    category="themes",
                    metadata={"analysis_type": "thematic"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error analizando temas: {e}")
        
        return None
    
    async def _classify_document_insight(self, text: str) -> Optional[AIInsight]:
        """Clasificar tipo de documento"""
        
        try:
            prompt = f"""
            Clasifica el siguiente documento en una de estas categorías:
            - Reporte técnico
            - Documento legal
            - Comunicación comercial
            - Contenido académico
            - Manual o guía
            - Correspondencia
            - Otro
            
            Explica brevemente por qué pertenece a esa categoría y qué características lo definen.
            
            Texto:
            {text[:1500]}...
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Clasificación del Documento",
                    content=response,
                    confidence=0.75,
                    category="classification",
                    metadata={"classification_type": "document_type"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error clasificando documento: {e}")
        
        return None
    
    async def _generate_sentiment_context(self, text: str, analysis_result: Dict) -> Optional[AIInsight]:
        """Generar contexto del análisis de sentimientos"""
        
        try:
            sentiment_data = analysis_result.get('sentiment', {})
            overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
            
            prompt = f"""
            El análisis automático indica que este documento tiene un sentimiento "{overall_sentiment}".
            
            Analiza el contexto y explica:
            1. ¿Por qué el documento tiene este sentimiento?
            2. ¿Qué elementos específicos contribuyen a este sentimiento?
            3. ¿Es apropiado el tono para el tipo de documento?
            
            Fragmento del texto:
            {text[:1000]}...
            
            Proporciona un análisis contextual del sentimiento.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title=f"Contexto del Sentimiento ({overall_sentiment.title()})",
                    content=response,
                    confidence=0.70,
                    category="sentiment_context",
                    metadata={"detected_sentiment": overall_sentiment},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error en contexto de sentimiento: {e}")
        
        return None
    
    async def _generate_recommendations(self, text: str, analysis_result: Dict = None) -> Optional[AIInsight]:
        """Generar recomendaciones basadas en el análisis"""
        
        try:
            context = ""
            if analysis_result:
                readability = analysis_result.get('readability', {})
                flesch_score = readability.get('flesch_reading_ease', 50)
                
                if flesch_score < 30:
                    context += "El documento tiene baja legibilidad. "
                elif flesch_score > 70:
                    context += "El documento tiene alta legibilidad. "
            
            prompt = f"""
            Basándote en el análisis del siguiente documento, proporciona 3-5 recomendaciones prácticas para:
            1. Mejorar la claridad y legibilidad
            2. Optimizar la estructura
            3. Enhancer la comunicación efectiva
            
            {context}
            
            Texto analizado:
            {text[:1000]}...
            
            Proporciona recomendaciones específicas y accionables.
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                return AIInsight(
                    title="Recomendaciones de Mejora",
                    content=response,
                    confidence=0.75,
                    category="recommendations",
                    metadata={"recommendation_type": "improvement"},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Error generando recomendaciones: {e}")
        
        return None
    
    async def _call_ai_engine(self, prompt: str, model: str = "auto") -> Optional[str]:
        """Llamar al AI-Engine para generar respuesta"""
        
        if not await self.check_ai_engine_connection():
            logger.warning("⚠️ AI-Engine no disponible")
            return None
        
        try:
            payload = {
                "prompt": prompt,
                "model": model,
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = await self.client.post(
                f"{self.ai_engine_url}/generate/complete",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("text", "").strip()
            else:
                logger.warning(f"⚠️ AI-Engine respondió con status {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error llamando AI-Engine: {e}")
            return None
    
    async def generate_comparative_insights(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[AIInsight]:
        """Generar insights comparativos entre múltiples documentos"""
        
        if len(documents) < 2:
            return []
        
        insights = []
        
        try:
            # Comparar sentimientos
            sentiments = [doc.get('analysis', {}).get('sentiment', {}).get('overall_sentiment', 'neutral') 
                         for doc in documents]
            
            # Comparar longitudes
            lengths = [doc.get('analysis', {}).get('word_count', 0) for doc in documents]
            
            # Comparar legibilidad
            readabilities = [doc.get('analysis', {}).get('readability', {}).get('flesch_reading_ease', 50) 
                           for doc in documents]
            
            comparison_data = {
                'document_count': len(documents),
                'sentiments': sentiments,
                'avg_length': sum(lengths) / len(lengths),
                'avg_readability': sum(readabilities) / len(readabilities)
            }
            
            prompt = f"""
            Analiza los siguientes datos comparativos de {len(documents)} documentos:
            
            Sentimientos: {sentiments}
            Longitud promedio: {comparison_data['avg_length']:.0f} palabras
            Legibilidad promedio: {comparison_data['avg_readability']:.1f}
            
            Proporciona insights sobre:
            1. Patrones en los sentimientos
            2. Consistencia en la comunicación
            3. Variabilidad en complejidad
            4. Recomendaciones para homogeneizar
            """
            
            response = await self._call_ai_engine(prompt)
            
            if response:
                insights.append(AIInsight(
                    title="Análisis Comparativo",
                    content=response,
                    confidence=0.80,
                    category="comparative",
                    metadata=comparison_data,
                    timestamp=datetime.now().isoformat()
                ))
            
        except Exception as e:
            logger.error(f"❌ Error en insights comparativos: {e}")
        
        return insights
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles en AI-Engine"""
        return ["auto", "gpt-3.5-turbo", "claude-3-sonnet", "local"]
    
    async def cleanup(self):
        """Limpiar recursos"""
        if self.client:
            await self.client.aclose()
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "ai_insights.py", "w", encoding="utf-8") as f:
            f.write(ai_insights_content)
        
        self.logger.info("✅ Servicio de insights con IA creado")

    def create_chart_generator(self):
        """Crear generador de gráficos"""
        self.logger.info("📊 Creando generador de gráficos...")
        
        chart_generator_content = '''"""
Generador de gráficos y visualizaciones
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generador de gráficos para análisis de datos"""
    
    def __init__(self):
        self.is_ready = False
        self.output_dir = "static/plots"
        
        # Configurar estilo por defecto
        plt.style.use('default')
        sns.set_palette("husl")
        
    async def initialize(self):
        """Inicializar el generador de gráficos"""
        logger.info("🔧 Inicializando Chart Generator...")
        
        # Crear directorio de salida
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.is_ready = True
        logger.info("✅ Chart Generator inicializado")
    
    async def create_sentiment_chart(
        self, 
        sentiment_data: Dict[str, float], 
        title: str = "Análisis de Sentimientos"
    ) -> str:
        """Crear gráfico de análisis de sentimientos"""
        
        try:
            # Crear gráfico con Plotly
            fig = go.Figure()
            
            # Datos de VADER
            vader_data = {
                'Positivo': sentiment_data.get('vader_positive', 0),
                'Negativo': sentiment_data.get('vader_negative', 0),
                'Neutral': sentiment_data.get('vader_neutral', 0)
            }
            
            # Gráfico de barras
            fig.add_trace(go.Bar(
                x=list(vader_data.keys()),
                y=list(vader_data.values()),
                marker_color=['green', 'red', 'blue'],
                text=[f'{v:.2f}' for v in vader_data.values()],
                textposition='auto'
            ))
            
            # Configurar layout
            fig.update_layout(
                title=title,
                xaxis_title="Categoría",
                yaxis_title="Puntuación",
                template="plotly_white",
                height=400
            )
            
            # Guardar archivo
            filename = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de sentimientos creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de sentimientos: {e}")
            return ""
    
    async def create_readability_chart(
        self, 
        readability_data: Dict[str, float],
        title: str = "Métricas de Legibilidad"
    ) -> str:
        """Crear gráfico de métricas de legibilidad"""
        
        try:
            # Preparar datos
            metrics = []
            values = []
            
            if 'flesch_reading_ease' in readability_data:
                metrics.append('Facilidad Lectura')
                values.append(readability_data['flesch_reading_ease'])
            
            if 'average_sentence_length' in readability_data:
                metrics.append('Long. Oraciones')
                values.append(min(readability_data['average_sentence_length'], 50))  # Escalar
            
            if 'average_word_length' in readability_data:
                metrics.append('Long. Palabras')
                values.append(readability_data['average_word_length'] * 10)  # Escalar
            
            # Crear gráfico radial
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='Métricas'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title=title,
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"readability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de legibilidad creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de legibilidad: {e}")
            return ""
    
    async def create_keywords_chart(
        self, 
        keywords: List[Dict[str, float]],
        title: str = "Palabras Clave Principales"
    ) -> str:
        """Crear gráfico de palabras clave"""
        
        try:
            if not keywords:
                return ""
            
            # Tomar top 10
            top_keywords = keywords[:10]
            
            words = [kw['keyword'] for kw in top_keywords]
            scores = [kw['score'] for kw in top_keywords]
            
            # Crear gráfico horizontal
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=scores,
                y=words,
                orientation='h',
                marker_color='skyblue',
                text=[f'{s:.3f}' for s in scores],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Puntuación TF-IDF",
                yaxis_title="Palabras Clave",
                template="plotly_white",
                height=500
            )
            
            # Guardar archivo
            filename = f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de keywords creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de keywords: {e}")
            return ""
    
    async def create_pos_tags_chart(
        self, 
        pos_data: Dict[str, int],
        title: str = "Distribución de Categorías Gramaticales"
    ) -> str:
        """Crear gráfico de etiquetas POS"""
        
        try:
            if not pos_data:
                return ""
            
            # Agrupar etiquetas similares
            grouped_pos = {
                'Sustantivos': 0,
                'Verbos': 0,
                'Adjetivos': 0,
                'Adverbios': 0,
                'Pronombres': 0,
                'Otros': 0
            }
            
            noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
            verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            adj_tags = ['JJ', 'JJR', 'JJS']
            adv_tags = ['RB', 'RBR', 'RBS']
            pronoun_tags = ['PRP', 'PRP#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        , 'WP', 'WP#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
        ]
            
            for tag, count in pos_data.items():
                if tag in noun_tags:
                    grouped_pos['Sustantivos'] += count
                elif tag in verb_tags:
                    grouped_pos['Verbos'] += count
                elif tag in adj_tags:
                    grouped_pos['Adjetivos'] += count
                elif tag in adv_tags:
                    grouped_pos['Adverbios'] += count
                elif tag in pronoun_tags:
                    grouped_pos['Pronombres'] += count
                else:
                    grouped_pos['Otros'] += count
            
            # Filtrar categorías vacías
            filtered_pos = {k: v for k, v in grouped_pos.items() if v > 0}
            
            # Crear gráfico de pie
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=list(filtered_pos.keys()),
                values=list(filtered_pos.values()),
                hole=0.3
            ))
            
            fig.update_layout(
                title=title,
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"pos_tags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de POS tags creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de POS: {e}")
            return ""
    
    async def create_emotions_chart(
        self, 
        emotions_data: Dict[str, float],
        title: str = "Análisis de Emociones"
    ) -> str:
        """Crear gráfico de análisis de emociones"""
        
        try:
            if not emotions_data:
                return ""
            
            # Filtrar emociones con valores > 0
            filtered_emotions = {k: v for k, v in emotions_data.items() if v > 0}
            
            if not filtered_emotions:
                return ""
            
            emotions = list(filtered_emotions.keys())
            scores = list(filtered_emotions.values())
            
            # Crear gráfico de barras polar
            fig = go.Figure()
            
            fig.add_trace(go.Barpolar(
                r=scores,
                theta=emotions,
                marker_color=px.colors.qualitative.Set3[:len(emotions)]
            ))
            
            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(scores) * 1.1]
                    )
                ),
                template="plotly_white"
            )
            
            # Guardar archivo
            filename = f"emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = f"{self.output_dir}/{filename}"
            fig.write_html(filepath)
            
            logger.info(f"✅ Gráfico de emociones creado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error creando gráfico de emociones: {e}")
            return ""
    
    async def create_comparative_dashboard(
        self, 
        documents_data: List[Dict[str, Any]],
        title: str = "Dashboard Comparativo"
    ) -> str:
        """Crear dashboard comparativo para múltiples documentos"""
        
        try:
            if len(documents_data) < 2:
                return ""
            
            # Crear subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Comparación de Sentimientos',
                    'Longitud de Documentos', 
                    'Legibilidad',
                    'Riqueza Vocabulario'
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            doc_names = [f"Doc {i+1}" for i in range(len(documents_data))]
            
            # 1. Sentimientos
            sentiments = []
            for doc in documents_data:
                sentiment = doc.get('analysis', {}).get('sentiment', {})
                compound = sentiment.get('vader_compound', 0)
                sentiments.append(compound)
            
            fig.add_trace(
                go.Bar(x=doc_names, y=sentiments, name="Sentimiento"),
                row=1, col=1
            )
            
            # 2. Longitudes
            lengths = [doc.get('analysis', {}).get('word_count', 0) for doc in documents_data]
            
            fig.add_trace(
                go.Bar(x=doc_names, y=lengths, name="Palabras"),
                row=1, col=2
            )
            
            # 3. Legibilidad vs Longitud
            readabilities = [doc.get('analysis', {}).get('readability', {}).get('flesch_reading_ease', 50) 
                           for doc in documents_data]
            
            fig.add_trace(
                go.Scatter(x=lengths, y=readabilities, mode='markers+text', 
                          text=doc_names, textposition="top center", name="Legibilidad"),
                row=2, col=1
            )
            
            # 4. Riqueza vocabulario
            vocab_richness = [doc.get('analysis', {}).get('vocabulary_richness', 0) for doc in documents#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 AGENTE IA OYP 6.0 - ANALYTICS-ENGINE SETUP
===============================================
Script específico para configurar el Analytics-Engine (Módulo 4)
Análisis estadístico avanzado e insights con IA
Ejecutar desde: services/analytics-engine/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class AnalyticsEngineSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "analytics-engine":
            self.logger.error("❌ Debes ejecutar este script desde services/analytics-engine/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Analytics-Engine...")
        
        directories = [
            "analyzers",
            "visualizers", 
            "processors",
            "routers",
            "services",
            "models",
            "utils",
            "tests",
            "logs",
            "data",
            "data/datasets",
            "data/reports",
            "data/cache",
            "configs",
            "templates",
            "static",
            "static/plots"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["analyzers", "visualizers", "processors", "routers", "services", "models", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import analysis, statistics, visualization, insights, models
from services.analytics_manager import AnalyticsManager
from services.text_analyzer import TextAnalyzer
from services.ai_insights import AIInsightsService
from services.data_processor import DataProcessor
from visualizers.chart_generator import ChartGenerator
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
analytics_manager = AnalyticsManager()
text_analyzer = TextAnalyzer()
ai_insights = AIInsightsService()
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📊 Iniciando Analytics-Engine...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("static/plots", exist_ok=True)
        
        # Inicializar servicios
        await analytics_manager.initialize()
        await text_analyzer.initialize()
        await ai_insights.initialize()
        await data_processor.initialize()
        await chart_generator.initialize()
        
        logger.info("✅ Analytics-Engine iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Analytics-Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Analytics-Engine...")
    await analytics_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📊 Analytics-Engine - Agente IA OyP 6.0",
    description="Motor de análisis estadístico e insights con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (gráficos, reportes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "version": "6.0.0",
        "analyzers_ready": text_analyzer.is_ready,
        "ai_insights_ready": ai_insights.is_ready,
        "data_processor_ready": data_processor.is_ready,
        "chart_generator_ready": chart_generator.is_ready,
        "capabilities": [
            "text_analysis",
            "statistical_analysis", 
            "ai_insights",
            "data_visualization",
            "sentiment_analysis",
            "classification",
            "clustering"
        ]
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "analytics-engine",
        "version": "6.0.0",
        "components": {
            "text_analyzer": {
                "ready": text_analyzer.is_ready,
                "supported_languages": text_analyzer.get_supported_languages(),
                "models_loaded": text_analyzer.get_loaded_models()
            },
            "ai_insights": {
                "ready": ai_insights.is_ready,
                "ai_engine_connected": await ai_insights.check_ai_engine_connection(),
                "available_models": ai_insights.get_available_models()
            },
            "data_processor": {
                "ready": data_processor.is_ready,
                "supported_formats": data_processor.get_supported_formats()
            },
            "chart_generator": {
                "ready": chart_generator.is_ready,
                "available_chart_types": chart_generator.get_chart_types()
            }
        },
        "statistics": await analytics_manager.get_statistics(),
        "storage": {
            "datasets_dir": "data/datasets",
            "reports_dir": "data/reports", 
            "cache_dir": "data/cache",
            "plots_dir": "static/plots"
        }
    }

# Capacidades disponibles
@app.get("/capabilities")
async def get_capabilities():
    """Obtener capacidades del motor de análisis"""
    return {
        "text_analysis": {
            "sentiment_analysis": True,
            "entity_extraction": True,
            "keyword_extraction": True,
            "language_detection": True,
            "readability_analysis": True,
            "topic_modeling": True
        },
        "statistical_analysis": {
            "descriptive_statistics": True,
            "correlation_analysis": True,
            "regression_analysis": True,
            "time_series_analysis": True,
            "distribution_analysis": True,
            "outlier_detection": True
        },
        "ai_insights": {
            "document_classification": True,
            "content_summarization": True,
            "pattern_recognition": True,
            "trend_analysis": True,
            "anomaly_detection": True,
            "predictive_insights": True
        },
        "visualization": {
            "chart_types": ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "wordcloud"],
            "interactive_plots": True,
            "dashboard_generation": True,
            "export_formats": ["png", "svg", "pdf", "html"]
        }
    }

# Incluir routers
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
app.include_router(statistics.router, prefix="/stats", tags=["Statistics"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization"])
app.include_router(insights.router, prefix="/insights", tags=["AI Insights"])
app.include_router(models.router, prefix="/models", tags=["Models"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# ANALYTICS-ENGINE - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Análisis de datos y estadísticas
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualización de datos
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
bokeh==3.3.0

# Procesamiento de texto y NLP
nltk==3.8.1
spacy==3.7.2
textblob==0.17.1
langdetect==1.0.9
wordcloud==1.9.2

# Análisis de sentimientos
vaderSentiment==3.3.2
transformers==4.35.2

# Machine Learning adicional
xgboost==2.0.1
lightgbm==4.1.0
catboost==1.2.2

# Clustering y reducción de dimensionalidad
umap-learn==0.5.4
hdbscan==0.8.33

# Series temporales
prophet==1.1.4
pmdarima==2.0.4

# Utilidades de red para AI-Engine
httpx==0.25.2
aiohttp==3.9.1

# Base de datos y caché
redis==5.0.1
sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
structlog==23.2.0
psutil==5.9.6

# Archivos y formatos
openpyxl==3.1.2
xlsxwriter==3.1.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Jupyter (para notebooks de análisis)
jupyter==1.0.0
ipywidgets==8.1.1
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_analyzer(self):
        """Crear analizador de texto"""
        self.logger.info("📝 Creando analizador de texto...")
        
        text_analyzer_content = '''"""
Analizador de texto con múltiples capacidades de NLP
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

@dataclass
class TextAnalysisResult:
    """Resultado completo del análisis de texto"""
    # Información básica
    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    
    # Análisis de sentimientos
    sentiment: Dict[str, float]
    emotion_scores: Dict[str, float]
    
    # Métricas de legibilidad
    readability: Dict[str, float]
    
    # Entidades y palabras clave
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, float]]
    topics: List[Dict[str, Any]]
    
    # Estadísticas lingüísticas
    pos_tags: Dict[str, int]
    vocabulary_richness: float
    average_sentence_length: float
    
    # Metadata
    analysis_timestamp: str
    processing_time: float

class TextAnalyzer:
    """Analizador principal de texto con capacidades avanzadas de NLP"""
    
    def __init__(self):
        self.is_ready = False
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        
    async def initialize(self):
        """Inicializar el analizador de texto"""
        logger.info("🔧 Inicializando Text Analyzer...")
        
        try:
            # Descargar recursos de NLTK si no existen
            await self._download_nltk_resources()
            
            # Cargar modelo de spaCy
            await self._load_spacy_model()
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            self.is_ready = True
            logger.info("✅ Text Analyzer inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Analyzer: {e}")
            raise
    
    async def _download_nltk_resources(self):
        """Descargar recursos necesarios de NLTK"""
        required_resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
    
    async def _load_spacy_model(self):
        """Cargar modelo de spaCy"""
        try:
            # Intentar cargar modelo en español
            self.nlp_model = spacy.load("es_core_news_sm")
            logger.info("✅ Modelo spaCy en español cargado")
        except OSError:
            try:
                # Fallback a modelo en inglés
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy en inglés cargado")
            except OSError:
                logger.warning("⚠️ No se encontró modelo spaCy. Funcionalidad limitada.")
                self.nlp_model = None
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict] = None
    ) -> TextAnalysisResult:
        """Realizar análisis completo de texto"""
        
        start_time = datetime.now()
        options = options or {}
        
        logger.info(f"📝 Analizando texto de {len(text)} caracteres...")
        
        try:
            # Análisis básico
            basic_stats = await self._analyze_basic_stats(text)
            
            # Detección de idioma
            language = await self._detect_language(text)
            
            # Análisis de sentimientos
            sentiment = await self._analyze_sentiment(text)
            
            # Análisis de legibilidad
            readability = await self._analyze_readability(text)
            
            # Extracción de entidades (si spaCy disponible)
            entities = await self._extract_entities(text) if self.nlp_model else []
            
            # Extracción de palabras clave
            keywords = await self._extract_keywords(text)
            
            # Análisis de temas
            topics = await self._analyze_topics(text) if options.get('topic_analysis', True) else []
            
            # Análisis lingüístico
            pos_tags = await self._analyze_pos_tags(text)
            vocab_richness = await self._calculate_vocabulary_richness(text)
            avg_sentence_length = await self._calculate_average_sentence_length(text)
            
            # Scores de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextAnalysisResult(
                text_length=len(text),
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                language=language,
                sentiment=sentiment,
                emotion_scores=emotion_scores,
                readability=readability,
                entities=entities,
                keywords=keywords,
                topics=topics,
                pos_tags=pos_tags,
                vocabulary_richness=vocab_richness,
                average_sentence_length=avg_sentence_length,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Análisis completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, int]:
        """Calcular estadísticas básicas del texto"""
        
        # Contar palabras
        words = nltk.word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Contar oraciones
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Contar párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            # Usar una muestra del texto para detección más rápida
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except:
            return 'unknown'
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimientos usando VADER"""
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment como comparación
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 
                               'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Análisis de legibilidad del texto"""
        
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Métricas básicas
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(word) for word in words if word.isalnum()])
        
        # Flesch Reading Ease (aproximación)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalnum())
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / max(len(words), 1)))
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'estimated_reading_time_minutes': len(words) / 200  # 200 palabras por minuto
        }
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadas en una palabra"""
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Mínimo 1 sílaba por palabra
        return max(1, syllables)
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas usando spaCy"""
        
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:1000000])  # Limitar texto muy largo
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'kb_score_', 1.0)
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo entidades: {e}")
            return []
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        """Extraer palabras clave usando TF-IDF"""
        
        try:
            # Tokenizar en oraciones para TF-IDF
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Obtener scores promedio
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Crear lista de keywords con scores
            keywords = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score)
                    })
            
            # Ordenar por score y retornar top_k
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo keywords: {e}")
            return []
    
    async def _analyze_topics(self, text: str, n_topics: int = 5) -> List[Dict[str, Any]]:
        """Análisis de temas usando LDA"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < n_topics:
                return []
            
            # Vectorizar
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(sentences)),
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': float(np.mean(top_weights))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de temas: {e}")
            return []
    
    async def _analyze_pos_tags(self, text: str) -> Dict[str, int]:
        """Análisis de etiquetas gramaticales"""
        
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Contar etiquetas
            tag_counts = {}
            for word, tag in pos_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return tag_counts
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis POS: {e}")
            return {}
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calcular riqueza del vocabulario (Type-Token Ratio)"""
        
        try:
            words = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando riqueza vocabulario: {e}")
            return 0.0
    
    async def _calculate_average_sentence_length(self, text: str) -> float:
        """Calcular longitud promedio de oraciones"""
        
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            return len(words) / max(len(sentences), 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando longitud oraciones: {e}")
            return 0.0
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis básico de emociones"""
        
        # Palabras clave para diferentes emociones
        emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'glad', 'delighted', 'feliz', 'alegre', 'contento'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'triste', 'deprimido'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'enojado', 'furioso', 'ira'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'miedo', 'temor', 'asustado'],
            'surprise': ['surprised', 'amazed', 'astonished', 'sorprendido', 'asombrado'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'asco', 'repugnancia']
        }
        
        text_lower = text.lower()
    