"""
Motor de Analytics - Servicio Principal COMPLETO
Puerto: 8003
Archivo: services/analytics-engine/src/main.py
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path

# Machine Learning y estadísticas
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import scipy.stats as stats

# Visualizaciones
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("⚠️ Plotly no disponible - visualizaciones limitadas")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================================
# MODELOS DE DATOS
# ===================================================================

class AnalysisRequest(BaseModel):
    data: Union[List[Dict], List[List], Dict]
    analysis_type: str = "descriptive"  # descriptive, correlation, clustering, regression
    target_column: Optional[str] = None
    parameters: Optional[Dict] = {}

class TextAnalyticsRequest(BaseModel):
    texts: List[str]
    analysis_type: str = "similarity"  # similarity, clustering, topics, keywords
    parameters: Optional[Dict] = {}

class VisualizationRequest(BaseModel):
    data: Union[List[Dict], List[List]]
    chart_type: str = "bar"  # bar, line, scatter, heatmap, histogram, box, pie
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    title: Optional[str] = "Analytics Chart"
    parameters: Optional[Dict] = {}

class StatisticalTestRequest(BaseModel):
    data: List[float]
    test_type: str = "normality"  # normality, ttest, correlation
    parameters: Optional[Dict] = {}

# ===================================================================
# APLICACIÓN FASTAPI
# ===================================================================

app = FastAPI(
    title="📊 Motor de Analytics - Agente IA OyP 6.0",
    description="Motor de análisis estadístico, machine learning y visualización de datos",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================
# CLASE PRINCIPAL DEL SERVICIO
# ===================================================================

class AnalyticsEngineService:
    """Motor de analytics con capacidades completas"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.models_cache = {}
        logger.info("✅ Analytics Engine Service inicializado")
    
    # ===================================================================
    # ANÁLISIS DESCRIPTIVO
    # ===================================================================
    
    async def descriptive_analysis(self, data: List[Dict]) -> Dict[str, Any]:
        """Análisis estadístico descriptivo completo"""
        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("Dataset vacío")
            
            result = {
                "dataset_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_types": df.dtypes.astype(str).to_dict(),
                    "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
                },
                "descriptive_stats": {},
                "missing_data": {},
                "data_quality": {},
                "correlations": {},
                "outliers": {},
                "distributions": {}
            }
            
            # Separar columnas por tipo
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Estadísticas para columnas numéricas
            if numeric_cols:
                numeric_stats = {}
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    numeric_stats[col] = {
                        "count": int(len(col_data)),
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "mode": float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                        "std": float(col_data.std()),
                        "var": float(col_data.var()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "q1": float(col_data.quantile(0.25)),
                        "q3": float(col_data.quantile(0.75)),
                        "skewness": float(col_data.skew()),
                        "kurtosis": float(col_data.kurtosis()),
                        "range": float(col_data.max() - col_data.min())
                    }
                
                result["descriptive_stats"]["numeric"] = numeric_stats
            
            # Estadísticas para columnas categóricas
            if categorical_cols:
                categorical_stats = {}
                for col in categorical_cols:
                    col_data = df[col].dropna()
                    value_counts = col_data.value_counts().to_dict()
                    
                    categorical_stats[col] = {
                        "unique_values": int(col_data.nunique()),
                        "most_frequent": col_data.mode().iloc[0] if not col_data.mode().empty else None,
                        "most_frequent_count": int(col_data.value_counts().iloc[0]) if len(value_counts) > 0 else 0,
                        "value_counts": dict(list(value_counts.items())[:10]),  # Top 10
                        "entropy": float(stats.entropy(list(value_counts.values()), base=2)) if value_counts else 0
                    }
                
                result["descriptive_stats"]["categorical"] = categorical_stats
            
            # Análisis de datos faltantes
            missing_data = df.isnull().sum()
            result["missing_data"] = {
                col: {
                    "count": int(count),
                    "percentage": float(count / len(df) * 100)
                }
                for col, count in missing_data.items()
                if count > 0
            }
            
            # Calidad de datos
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            
            result["data_quality"] = {
                "completeness": float((1 - missing_cells / total_cells) * 100),
                "duplicate_rows": int(duplicate_rows),
                "duplicate_percentage": float(duplicate_rows / len(df) * 100),
                "total_cells": total_cells,
                "missing_cells": int(missing_cells)
            }
            
            # Matriz de correlación (solo numéricas)
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                result["correlations"] = {
                    "matrix": corr_matrix.to_dict(),
                    "strong_correlations": self.find_strong_correlations(corr_matrix),
                    "correlation_summary": {
                        "max_correlation": float(corr_matrix.abs().max().max()),
                        "min_correlation": float(corr_matrix.abs().min().min()),
                        "avg_correlation": float(corr_matrix.abs().mean().mean())
                    }
                }
            
            # Detección de outliers
            if numeric_cols:
                outliers = {}
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    outlier_count = outlier_mask.sum()
                    
                    outliers[col] = {
                        "count": int(outlier_count),
                        "percentage": float(outlier_count / len(col_data) * 100),
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                        "outlier_values": col_data[outlier_mask].tolist()[:10]  # Primeros 10
                    }
                
                result["outliers"] = outliers
            
            # Análisis de distribuciones
            if numeric_cols:
                distributions = {}
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    
                    # Test de normalidad (Shapiro-Wilk para muestras pequeñas, Anderson para grandes)
                    if len(col_data) <= 5000:
                        stat, p_value = stats.shapiro(col_data)
                        test_name = "shapiro_wilk"
                    else:
                        stat, crit_vals, sig_level = stats.anderson(col_data, dist='norm')
                        p_value = 0.05 if stat > crit_vals[2] else 0.1  # Aproximación
                        test_name = "anderson_darling"
                    
                    distributions[col] = {
                        "normality_test": {
                            "test": test_name,
                            "statistic": float(stat),
                            "p_value": float(p_value),
                            "is_normal": p_value > 0.05
                        },
                        "distribution_type": "normal" if p_value > 0.05 else "non_normal"
                    }
                
                result["distributions"] = distributions
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis descriptivo: {e}")
            raise HTTPException(status_code=500, detail=f"Error análisis: {str(e)}")
    
    def find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Encontrar correlaciones fuertes"""
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corr.append({
                        "variable_1": corr_matrix.columns[i],
                        "variable_2": corr_matrix.columns[j],
                        "correlation": float(corr_value),
                        "strength": "very_strong" if abs(corr_value) >= 0.9 else "strong",
                        "direction": "positive" if corr_value > 0 else "negative"
                    })
        return strong_corr
    
    # ===================================================================
    # CLUSTERING
    # ===================================================================
    
    async def clustering_analysis(self, data: List[Dict], parameters: Dict = {}) -> Dict[str, Any]:
        """Análisis de clustering con K-Means"""
        try:
            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                raise ValueError("Se necesitan al menos 2 columnas numéricas para clustering")
            
            # Preparar datos
            X = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Normalizar datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determinar número óptimo de clusters
            max_clusters = min(parameters.get('max_clusters', 10), len(X) - 1)
            inertias = []
            silhouette_scores = []
            
            K_range = range(2, max_clusters + 1)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                inertias.append(kmeans.inertia_)
                
                if len(set(cluster_labels)) > 1:  # Solo si hay más de un cluster
                    sil_score = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)
            
            # Seleccionar mejor k
            if silhouette_scores:
                optimal_k = K_range[np.argmax(silhouette_scores)]
            else:
                optimal_k = 3
            
            # Clustering final
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = final_kmeans.fit_predict(X_scaled)
            
            # Agregar clusters al dataframe original
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = cluster_labels
            
            # Análisis por cluster
            cluster_analysis = {}
            for cluster_id in range(optimal_k):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                cluster_stats = cluster_data[numeric_cols].describe().to_dict()
                
                cluster_analysis[f"cluster_{cluster_id}"] = {
                    "size": len(cluster_data),
                    "percentage": float(len(cluster_data) / len(df) * 100),
                    "centroid": cluster_data[numeric_cols].mean().to_dict(),
                    "statistics": cluster_stats,
                    "characteristics": self.describe_cluster_characteristics(cluster_data[numeric_cols], df[numeric_cols])
                }
            
            result = {
                "optimal_clusters": optimal_k,
                "cluster_analysis": cluster_analysis,
                "model_metrics": {
                    "inertia": float(final_kmeans.inertia_),
                    "silhouette_score": float(silhouette_score(X_scaled, cluster_labels))
                },
                "optimization_data": {
                    "k_values": list(K_range),
                    "inertias": inertias,
                    "silhouette_scores": silhouette_scores
                },
                "clustered_data": df_with_clusters.to_dict('records'),
                "feature_importance": {
                    col: float(np.std(X_scaled[:, i]))
                    for i, col in enumerate(numeric_cols)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en clustering: {e}")
            raise HTTPException(status_code=500, detail=f"Error clustering: {str(e)}")
    
    def describe_cluster_characteristics(self, cluster_data: pd.DataFrame, overall_data: pd.DataFrame) -> Dict[str, Any]:
        """Describir características distintivas de un cluster"""
        characteristics = {}
        
        for col in cluster_data.columns:
            cluster_mean = cluster_data[col].mean()
            overall_mean = overall_data[col].mean()
            difference = cluster_mean - overall_mean
            
            characteristics[col] = {
                "cluster_mean": float(cluster_mean),
                "overall_mean": float(overall_mean),
                "difference": float(difference),
                "relative_difference": float(difference / overall_mean * 100) if overall_mean != 0 else 0,
                "interpretation": "above_average" if difference > 0 else "below_average" if difference < 0 else "average"
            }
        
        return characteristics
    
    # ===================================================================
    # ANÁLISIS DE TEXTO
    # ===================================================================
    
    async def text_analytics(self, request: TextAnalyticsRequest) -> Dict[str, Any]:
        """Análisis de texto avanzado"""
        try:
            texts = request.texts
            analysis_type = request.analysis_type
            
            if not texts:
                raise ValueError("Lista de textos vacía")
            
            result = {
                "analysis_type": analysis_type,
                "text_count": len(texts),
                "timestamp": datetime.now().isoformat()
            }
            
            if analysis_type == "similarity":
                result.update(await self.text_similarity_analysis(texts))
            elif analysis_type == "clustering":
                result.update(await self.text_clustering_analysis(texts))
            elif analysis_type == "keywords":
                result.update(await self.text_keyword_extraction(texts))
            elif analysis_type == "topics":
                result.update(await self.text_topic_analysis(texts))
            else:
                # Análisis completo
                result.update({
                    "similarity": await self.text_similarity_analysis(texts),
                    "keywords": await self.text_keyword_extraction(texts),
                    "basic_stats": self.basic_text_statistics(texts)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise HTTPException(status_code=500, detail=f"Error análisis texto: {str(e)}")
    
    async def text_similarity_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Análisis de similitud entre textos"""
        try:
            # Vectorización TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Matriz de similitud
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Encontrar pares similares
            similar_pairs = []
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    similarity = similarity_matrix[i][j]
                    if similarity > 0.3:
                        similar_pairs.append({
                            "text_1_index": i,
                            "text_2_index": j,
                            "similarity_score": float(similarity),
                            "text_1_preview": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                            "text_2_preview": texts[j][:100] + "..." if len(texts[j]) > 100 else texts[j]
                        })
            
            similar_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return {
                "similarity_matrix": similarity_matrix.tolist(),
                "similar_pairs": similar_pairs[:10],
                "similarity_statistics": {
                    "avg_similarity": float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
                    "max_similarity": float(np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
                    "min_similarity": float(np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error en similitud: {e}")
            return {"error": str(e)}
    
    async def text_clustering_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Clustering de textos"""
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            max_clusters = min(5, len(texts) // 2)
            if max_clusters < 2:
                return {"error": "Insuficientes textos para clustering"}
            
            optimal_k = min(3, max_clusters)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix.toarray())
            
            cluster_info = {}
            for cluster_id in range(optimal_k):
                cluster_texts = [texts[i] for i, c in enumerate(clusters) if c == cluster_id]
                
                cluster_info[f"cluster_{cluster_id}"] = {
                    "size": len(cluster_texts),
                    "percentage": float(len(cluster_texts) / len(texts) * 100),
                    "sample_texts": cluster_texts[:3],
                    "representative_words": self.get_cluster_keywords(cluster_texts)
                }
            
            return {
                "clusters": optimal_k,
                "cluster_info": cluster_info,
                "clustered_texts": [{"text": text, "cluster": int(cluster)} for text, cluster in zip(texts, clusters)]
            }
            
        except Exception as e:
            logger.error(f"❌ Error en clustering texto: {e}")
            return {"error": str(e)}
    
    async def text_keyword_extraction(self, texts: List[str]) -> Dict[str, Any]:
        """Extracción de palabras clave"""
        try:
            # TF-IDF para keywords
            vectorizer = TfidfVectorizer(max_features=50, stop_words=None, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            keywords = [
                {"keyword": feature_names[i], "score": float(mean_scores[i])}
                for i in range(len(feature_names))
            ]
            
            keywords.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                "top_keywords": keywords[:20],
                "total_keywords": len(keywords),
                "extraction_method": "TF-IDF"
            }
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo keywords: {e}")
            return {"error": str(e)}
    
    async def text_topic_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Análisis de tópicos básico"""
        try:
            # Análisis simple por frecuencia de palabras
            all_words = []
            for text in texts:
                words = text.lower().split()
                filtered_words = [w for w in words if len(w) > 3]
                all_words.extend(filtered_words)
            
            # Contar frecuencias
            word_freq = {}
            for word in all_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "main_topics": sorted_words[:20],
                "total_unique_words": len(word_freq),
                "total_words": len(all_words),
                "analysis_method": "frequency_based"
            }
            
        except Exception as e:
            logger.error(f"❌ Error en tópicos: {e}")
            return {"error": str(e)}
    
    def get_cluster_keywords(self, cluster_texts: List[str], max_words: int = 5) -> List[str]:
        """Obtener palabras representativas de un cluster"""
        try:
            if not cluster_texts:
                return []
            
            combined_text = " ".join(cluster_texts).lower()
            words = combined_text.split()
            
            # Filtrar palabras comunes
            stop_words = {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo'}
            filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
            
            # Contar frecuencias
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Obtener top palabras
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:max_words]]
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo keywords cluster: {e}")
            return []
    
    def basic_text_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Estadísticas básicas de textos"""
        try:
            lengths = [len(text) for text in texts]
            word_counts = [len(text.split()) for text in texts]
            
            return {
                "total_texts": len(texts),
                "avg_length_chars": float(np.mean(lengths)),
                "avg_length_words": float(np.mean(word_counts)),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "total_characters": sum(lengths),
                "total_words": sum(word_counts)
            }
            
        except Exception as e:
            logger.error(f"❌ Error estadísticas texto: {e}")
            return {}
    
    # ===================================================================
    # VISUALIZACIONES
    # ===================================================================
    
    async def create_visualization(self, request: VisualizationRequest) -> Dict[str, Any]:
        """Crear visualizaciones con Plotly"""
        try:
            if not PLOTLY_AVAILABLE:
                return {"error": "Plotly no disponible", "chart": None}
            
            df = pd.DataFrame(request.data)
            chart_type = request.chart_type
            title = request.title
            
            fig = None
            
            if chart_type == "bar":
                if request.x_column and request.y_column:
                    fig = px.bar(df, x=request.x_column, y=request.y_column, title=title)
                else:
                    # Auto-detectar columnas
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if categorical_cols and numeric_cols:
                        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], title=title)
            
            elif chart_type == "line":
                if request.x_column and request.y_column:
                    fig = px.line(df, x=request.x_column, y=request.y_column, title=title)
            
            elif chart_type == "scatter":
                if request.x_column and request.y_column:
                    fig = px.scatter(df, x=request.x_column, y=request.y_column, title=title)
            
            elif chart_type == "histogram":
                if request.x_column:
                    fig = px.histogram(df, x=request.x_column, title=title)
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        fig = px.histogram(df, x=numeric_cols[0], title=title)
            
            elif chart_type == "box":
                if request.y_column:
                    fig = px.box(df, y=request.y_column, title=title)
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        fig = px.box(df, y=numeric_cols[0], title=title)
            
            elif chart_type == "heatmap":
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(corr_matrix, 
                                   labels=dict(x="Variables", y="Variables", color="Correlación"),
                                   title="Matriz de Correlación")
                else:
                    return {"error": "No hay columnas numéricas para heatmap"}
            
            elif chart_type == "pie":
                if request.x_column and request.y_column:
                    fig = px.pie(df, names=request.x_column, values=request.y_column, title=title)
            
            else:
                return {"error": f"Tipo de gráfico '{chart_type}' no soportado"}
            
            if fig:
                chart_json = json.loads(fig.to_json())
                return {
                    "chart_type": chart_type,
                    "title": title,
                    "chart_data": chart_json,
                    "success": True
                }
            else:
                return {"error": "No se pudo generar el gráfico con los datos proporcionados"}
            
        except Exception as e:
            logger.error(f"❌ Error creando visualización: {e}")
            return {"error": str(e), "chart": None}
    
    # ===================================================================
    # TESTS ESTADÍSTICOS
    # ===================================================================
    
    async def statistical_tests(self, request: StatisticalTestRequest) -> Dict[str, Any]:
        """Realizar tests estadísticos"""
        try:
            data = np.array(request.data)
            test_type = request.test_type
            
            results = {
                "test_type": test_type,
                "sample_size": len(data),
                "timestamp": datetime.now().isoformat()
            }
            
            if test_type == "normality":
                # Test de normalidad
                if len(data) <= 5000:
                    statistic, p_value = stats.shapiro(data)
                    test_name = "Shapiro-Wilk"
                else:
                    statistic, critical_values, significance_level = stats.anderson(data, dist='norm')
                    p_value = 0.05 if statistic > critical_values[2] else 0.1
                    test_name = "Anderson-Darling"
                
                results.update({
                    "test_name": test_name,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05,
                    "interpretation": "Los datos siguen una distribución normal" if p_value > 0.05 else "Los datos NO siguen una distribución normal"
                })
            
            elif test_type == "descriptive":
                # Estadísticas descriptivas completas
                results.update({
                    "mean": float(np.mean(data)),
                    "median": float(np.median(data)),
                    "mode": float(stats.mode(data)[0][0]) if len(stats.mode(data)[0]) > 0 else None,
                    "std": float(np.std(data, ddof=1)),
                    "variance": float(np.var(data, ddof=1)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "range": float(np.max(data) - np.min(data)),
                    "q1": float(np.percentile(data, 25)),
                    "q3": float(np.percentile(data, 75)),
                    "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
                    "skewness": float(stats.skew(data)),
                    "kurtosis": float(stats.kurtosis(data)),
                    "coefficient_variation": float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else None
                })
            
            elif test_type == "outliers":
                # Detección de outliers
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                results.update({
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(len(outliers) / len(data) * 100),
                    "outliers": outliers.tolist(),
                    "bounds": {
                        "lower": float(lower_bound),
                        "upper": float(upper_bound)
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en test estadístico: {e}")
            raise HTTPException(status_code=500, detail=f"Error en test: {str(e)}")

# ===================================================================
# INSTANCIA GLOBAL DEL SERVICIO
# ===================================================================

analytics_service = AnalyticsEngineService()

# ===================================================================
# ENDPOINTS DE LA API
# ===================================================================

@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "message": "📊 Bienvenido al Motor de Analytics - Agente IA OyP 6.0",
        "service": "analytics-engine",
        "version": "6.0.0",
        "status": "active",
        "capabilities": [
            "descriptive_analysis",
            "correlation_analysis",
            "clustering_analysis",
            "text_analytics",
            "statistical_tests",
            "data_visualization",
            "outlier_detection",
            "regression_analysis"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check del servicio"""
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "port": 8003,
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "pandas": True,
            "numpy": True,
            "scikit_learn": True,
            "scipy": True,
            "plotly": PLOTLY_AVAILABLE
        },
        "cached_analyses": len(analytics_service.analysis_cache)
    }

@app.get("/info")
async def service_info():
    """Información detallada del servicio"""
    return {
        "name": "analytics-engine",
        "description": "Motor de análisis estadístico, machine learning y visualización",
        "port": 8003,
        "version": "6.0.0",
        "endpoints": {
            "GET /": "Información del servicio",
            "GET /health": "Health check",
            "GET /info": "Información detallada",
            "POST /analyze": "Análisis de datos principales",
            "POST /text_analytics": "Análisis de texto",
            "POST /visualize": "Crear visualizaciones",
            "POST /statistical_test": "Tests estadísticos",
            "POST /clustering": "Análisis de clustering"
        }
    }

@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """Endpoint principal para análisis de datos"""
    analysis_type = request.analysis_type
    
    if analysis_type == "descriptive":
        return await analytics_service.descriptive_analysis(request.data)
    elif analysis_type == "clustering":
        return await analytics_service.clustering_analysis(request.data, request.parameters or {})
    else:
        raise HTTPException(status_code=400, detail=f"Tipo de análisis '{analysis_type}' no soportado")

@app.post("/text_analytics")
async def text_analytics_endpoint(request: TextAnalyticsRequest):
    """Análisis de texto avanzado"""
    return await analytics_service.text_analytics(request)

@app.post("/visualize")
async def create_visualization_endpoint(request: VisualizationRequest):
    """Crear visualizaciones"""
    return await analytics_service.create_visualization(request)

@app.post("/statistical_test")
async def statistical_test_endpoint(request: StatisticalTestRequest):
    """Realizar tests estadísticos"""
    return await analytics_service.statistical_tests(request)

@app.post("/clustering")
async def clustering_endpoint(data: List[Dict], parameters: Dict = {}):
    """Análisis de clustering específico"""
    return await analytics_service.clustering_analysis(data, parameters)

@app.post("/correlation")
async def correlation_analysis(data: List[Dict]):
    """Análisis de correlación específico"""
    try:
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 columnas numéricas")
        
        corr_matrix = df[numeric_cols].corr()
        strong_correlations = analytics_service.find_strong_correlations(corr_matrix)
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "numeric_columns": numeric_cols,
            "correlation_summary": {
                "max_correlation": float(corr_matrix.abs().max().max()),
                "avg_correlation": float(corr_matrix.abs().mean().mean()),
                "total_pairs": len(strong_correlations)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error en correlación: {e}")
        raise HTTPException(status_code=500, detail=f"Error correlación: {str(e)}")

@app.get("/statistics")
async def get_service_statistics():
    """Estadísticas del servicio"""
    return {
        "service_stats": {
            "cached_analyses": len(analytics_service.analysis_cache),
            "models_cached": len(analytics_service.models_cache)
        },
        "capabilities": {
            "plotly_available": PLOTLY_AVAILABLE,
            "machine_learning": True,
            "statistical_tests": True,
            "text_processing": True
        },
        "supported_analyses": [
            "descriptive_statistics",
            "correlation_analysis",
            "clustering_kmeans", 
            "text_similarity",
            "keyword_extraction",
            "outlier_detection",
            "normality_tests",
            "data_visualization"
        ]
    }

# ===================================================================
# INICIALIZACIÓN DEL SERVICIO
# ===================================================================

if __name__ == "__main__":
    logger.info("🚀 Iniciando Analytics Engine Service...")
    logger.info("📍 Puerto: 8003")
    logger.info("📖 Documentación: http://localhost:8003/docs")
    logger.info(f"📊 Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
    logger.info("🔬 Capacidades: estadísticas, clustering, análisis texto, visualizaciones")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )