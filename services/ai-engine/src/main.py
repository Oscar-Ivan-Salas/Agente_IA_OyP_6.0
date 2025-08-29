"""
Motor de IA - Servicio Principal COMPLETO
Puerto: 8001
Archivo: services/ai-engine/src/main.py
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import re
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================================
# MODELOS DE DATOS
# ===================================================================

class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "complete"  # complete, sentiment, entities, summary
    language: str = "es"

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"
    context: Optional[Dict] = {}

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 50

class ClassificationRequest(BaseModel):
    text: str
    categories: List[str]

# ===================================================================
# APLICACIÓN FASTAPI
# ===================================================================

app = FastAPI(
    title="🤖 Motor de IA - Agente IA OyP 6.0",
    description="Motor de inteligencia artificial con análisis de texto, chat y procesamiento NLP",
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

class AIEngineService:
    """Motor de IA con capacidades completas"""
    
    def __init__(self):
        self.conversations = {}
        self.analysis_cache = {}
        logger.info("✅ AI Engine Service inicializado")
    
    # ===================================================================
    # ANÁLISIS DE TEXTO
    # ===================================================================
    
    async def analyze_text_complete(self, request: TextAnalysisRequest) -> Dict[str, Any]:
        """Análisis completo de texto con IA"""
        try:
            text = request.text
            results = {
                "text": text,
                "analysis_type": request.analysis_type,
                "language": request.language,
                "timestamp": datetime.now().isoformat(),
                "results": {}
            }
            
            # Análisis estadístico básico
            results["results"]["statistics"] = self.get_text_statistics(text)
            
            # Análisis de sentimientos
            if request.analysis_type in ["complete", "sentiment"]:
                results["results"]["sentiment"] = self.analyze_sentiment(text)
            
            # Extracción de entidades
            if request.analysis_type in ["complete", "entities"]:
                results["results"]["entities"] = self.extract_entities(text)
            
            # Resumen automático
            if request.analysis_type in ["complete", "summary"]:
                results["results"]["summary"] = self.generate_summary(text)
            
            # Palabras clave
            if request.analysis_type == "complete":
                results["results"]["keywords"] = self.extract_keywords(text)
                results["results"]["language_detection"] = self.detect_language(text)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de texto: {e}")
            raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Análisis de sentimientos avanzado"""
        try:
            # Palabras positivas y negativas en español
            positive_words = [
                'excelente', 'bueno', 'fantástico', 'genial', 'perfecto', 'maravilloso',
                'increíble', 'estupendo', 'magnífico', 'extraordinario', 'brillante',
                'feliz', 'alegre', 'contento', 'satisfecho', 'encantado', 'gratificante',
                'positivo', 'optimista', 'esperanzador', 'motivador', 'inspirador'
            ]
            
            negative_words = [
                'terrible', 'malo', 'horrible', 'pésimo', 'awful', 'desastroso',
                'lamentable', 'deplorable', 'abominable', 'espantoso', 'nefasto',
                'triste', 'deprimido', 'melancólico', 'desanimado', 'preocupante',
                'negativo', 'pesimista', 'desesperanzador', 'frustrante', 'decepcionante'
            ]
            
            neutral_words = [
                'normal', 'regular', 'común', 'estándar', 'típico', 'habitual',
                'ordinario', 'promedio', 'medio', 'neutro', 'equilibrado'
            ]
            
            # Análisis intensificadores
            intensifiers = ['muy', 'sumamente', 'extremadamente', 'completamente', 'totalmente']
            
            text_lower = text.lower()
            words = text_lower.split()
            
            # Contar palabras con contexto
            positive_score = 0
            negative_score = 0
            neutral_score = 0
            
            for i, word in enumerate(words):
                multiplier = 1
                
                # Verificar intensificadores
                if i > 0 and words[i-1] in intensifiers:
                    multiplier = 1.5
                
                if word in positive_words:
                    positive_score += multiplier
                elif word in negative_words:
                    negative_score += multiplier
                elif word in neutral_words:
                    neutral_score += multiplier
            
            # Calcular sentimiento
            total_score = positive_score + negative_score + neutral_score
            
            if total_score == 0:
                sentiment = "NEUTRAL"
                confidence = 0.5
            else:
                if positive_score > negative_score and positive_score > neutral_score:
                    sentiment = "POSITIVE"
                    confidence = min(0.95, 0.6 + (positive_score / total_score) * 0.35)
                elif negative_score > positive_score and negative_score > neutral_score:
                    sentiment = "NEGATIVE"
                    confidence = min(0.95, 0.6 + (negative_score / total_score) * 0.35)
                else:
                    sentiment = "NEUTRAL"
                    confidence = 0.6 + (neutral_score / total_score) * 0.2
            
            # Análisis adicional - patrones de puntuación
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            
            # Ajustar confianza basado en patrones
            if exclamation_count > 2:
                confidence = min(0.98, confidence + 0.1)
            if caps_ratio > 0.3:
                confidence = min(0.98, confidence + 0.05)
            
            return {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "scores": {
                    "positive": float(positive_score),
                    "negative": float(negative_score),
                    "neutral": float(neutral_score)
                },
                "indicators": {
                    "exclamations": exclamation_count,
                    "questions": question_count,
                    "caps_ratio": float(caps_ratio)
                },
                "method": "rule_based_advanced"
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de sentimientos: {e}")
            return {"sentiment": "NEUTRAL", "confidence": 0.0, "error": str(e)}
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extracción de entidades nombradas"""
        try:
            entities = []
            
            # Patrones de expresiones regulares para entidades
            patterns = {
                "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "PHONE": r'\b(?:\+?34[-.\s]?)?[6-9]\d{8}\b',
                "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                "TIME": r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b',
                "MONEY": r'\b\d+(?:\.\d{2})?\s?(?:€|EUR|euros?|dólares?|\$)\b',
                "PERCENTAGE": r'\b\d+(?:\.\d+)?%\b',
                "URL": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?',
                "DNI": r'\b\d{8}[A-Z]\b',
                "POSTAL_CODE": r'\b\d{5}\b'
            }
            
            for entity_type, pattern in patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        "text": match.group(),
                        "label": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.9
                    })
            
            # Entidades por capitalización (nombres propios)
            words = text.split()
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2 and word.isalpha():
                    # Verificar que no sea inicio de oración
                    if i > 0 or (i == 0 and len(words) > 1):
                        entities.append({
                            "text": word,
                            "label": "PROPER_NAME",
                            "confidence": 0.7,
                            "method": "capitalization"
                        })
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo entidades: {e}")
            return []
    
    def generate_summary(self, text: str) -> Dict[str, Any]:
        """Generar resumen automático"""
        try:
            sentences = text.split('. ')
            
            if len(sentences) <= 3:
                return {
                    "summary": text,
                    "method": "no_summary_needed",
                    "original_length": len(text),
                    "summary_length": len(text),
                    "compression_ratio": 1.0
                }
            
            # Algoritmo de resumen extractivo por posición y palabras clave
            sentence_scores = {}
            
            # Palabras clave importantes
            keywords = self.extract_keywords(text)[:10]
            
            for i, sentence in enumerate(sentences):
                score = 0
                words = sentence.lower().split()
                
                # Puntuación por posición (primeras y últimas oraciones más importantes)
                if i < 2:  # Primeras oraciones
                    score += 2
                elif i >= len(sentences) - 2:  # Últimas oraciones
                    score += 1.5
                
                # Puntuación por palabras clave
                for keyword in keywords:
                    if keyword.lower() in sentence.lower():
                        score += 1
                
                # Puntuación por longitud (evitar oraciones muy cortas o muy largas)
                if 10 <= len(words) <= 30:
                    score += 1
                
                sentence_scores[i] = score
            
            # Seleccionar mejores oraciones
            sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            selected_indices = sorted([idx for idx, score in sorted_sentences[:3]])
            
            summary_sentences = [sentences[i] for i in selected_indices]
            summary = '. '.join(summary_sentences)
            
            if not summary.endswith('.'):
                summary += '.'
            
            return {
                "summary": summary,
                "method": "extractive_keywords",
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text),
                "selected_sentences": len(selected_indices),
                "total_sentences": len(sentences)
            }
            
        except Exception as e:
            logger.error(f"❌ Error generando resumen: {e}")
            return {"summary": text[:200] + "...", "method": "truncation", "error": str(e)}
    
    def extract_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """Extracción de palabras clave por frecuencia TF"""
        try:
            # Limpiar texto
            text_lower = text.lower()
            words = re.findall(r'\b[a-záéíóúñü]{3,}\b', text_lower)
            
            # Palabras vacías en español
            stop_words = {
                'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su',
                'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'pero', 'sus', 'ha', 'me',
                'si', 'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 'todo', 'esta', 'ser', 'más', 'muy',
                'como', 'también', 'puede', 'donde', 'cada', 'fue', 'son', 'han', 'desde', 'hasta', 'cual',
                'otros', 'otras', 'otro', 'otra', 'mismo', 'misma', 'bien', 'ser', 'estar', 'tener', 'hacer'
            }
            
            # Filtrar palabras vacías
            filtered_words = [word for word in words if word not in stop_words]
            
            # Contar frecuencias
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Ordenar por frecuencia
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [word for word, freq in sorted_words[:max_keywords]]
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo keywords: {e}")
            return []
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detección de idioma por análisis de palabras comunes"""
        try:
            common_spanish = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da']
            common_english = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on']
            common_french = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce']
            
            text_lower = text.lower()
            words = text_lower.split()
            
            spanish_count = sum(1 for word in common_spanish if word in words)
            english_count = sum(1 for word in common_english if word in words)
            french_count = sum(1 for word in common_french if word in words)
            
            total_words = len(words)
            
            scores = {
                'es': spanish_count / max(total_words, 1),
                'en': english_count / max(total_words, 1),
                'fr': french_count / max(total_words, 1)
            }
            
            detected_lang = max(scores.items(), key=lambda x: x[1])
            
            return {
                "detected_language": detected_lang[0],
                "confidence": float(detected_lang[1]),
                "scores": {lang: float(score) for lang, score in scores.items()},
                "method": "common_words"
            }
            
        except Exception as e:
            logger.error(f"❌ Error detectando idioma: {e}")
            return {"detected_language": "unknown", "confidence": 0.0}
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Estadísticas completas del texto"""
        try:
            words = text.split()
            sentences = text.split('.')
            paragraphs = text.split('\n\n')
            
            # Estadísticas básicas
            char_count = len(text)
            char_count_no_spaces = len(text.replace(' ', ''))
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            paragraph_count = len([p for p in paragraphs if p.strip()])
            
            # Estadísticas avanzadas
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Análisis de complejidad
            long_words = len([w for w in words if len(w) > 6])
            complexity_score = long_words / max(word_count, 1)
            
            # Tiempo de lectura estimado
            reading_time_minutes = word_count / 200  # 200 WPM promedio
            
            return {
                "character_count": char_count,
                "character_count_no_spaces": char_count_no_spaces,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_word_length": float(avg_word_length),
                "avg_sentence_length": float(avg_sentence_length),
                "long_words_count": long_words,
                "complexity_score": float(complexity_score),
                "reading_time_minutes": float(reading_time_minutes),
                "readability": "Alta" if complexity_score < 0.3 else "Media" if complexity_score < 0.6 else "Baja"
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculando estadísticas: {e}")
            return {}
    
    # ===================================================================
    # CHAT CONVERSACIONAL
    # ===================================================================
    
    async def process_chat(self, request: ChatRequest) -> Dict[str, Any]:
        """Procesar mensaje de chat con IA"""
        try:
            message = request.message
            conversation_id = request.conversation_id
            
            # Mantener historial de conversación
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            # Agregar mensaje del usuario
            self.conversations[conversation_id].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generar respuesta basada en análisis del mensaje
            response = await self.generate_chat_response(message, conversation_id)
            
            # Agregar respuesta del asistente
            self.conversations[conversation_id].append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "response": response,
                "conversation_id": conversation_id,
                "message_count": len(self.conversations[conversation_id]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en chat: {e}")
            raise HTTPException(status_code=500, detail=f"Error en chat: {str(e)}")
    
    async def generate_chat_response(self, message: str, conversation_id: str) -> str:
        """Generar respuesta de chat inteligente"""
        try:
            message_lower = message.lower()
            
            # Patrones de respuesta basados en análisis del mensaje
            if any(word in message_lower for word in ['hola', 'buenos días', 'buenas tardes', 'saludar']):
                return "¡Hola! Soy tu asistente de IA. Puedo ayudarte con análisis de texto, responder preguntas y procesar información. ¿En qué puedo ayudarte hoy?"
            
            elif any(word in message_lower for word in ['analizar', 'análisis', 'estudiar']):
                return "Perfecto, puedo hacer análisis de texto avanzado incluyendo sentimientos, extracción de entidades, resúmenes automáticos y estadísticas. ¿Qué texto te gustaría que analice?"
            
            elif any(word in message_lower for word in ['ayuda', 'help', 'qué puedes hacer', 'funciones']):
                return """Estas son mis principales capacidades:
                
🤖 **Análisis de Texto:**
• Análisis de sentimientos
• Extracción de entidades (emails, teléfonos, fechas, etc.)
• Generación de resúmenes automáticos
• Extracción de palabras clave
• Estadísticas textuales completas

💬 **Chat Inteligente:**
• Conversación natural
• Análisis de consultas
• Respuestas contextuales

¿Con qué te gustaría empezar?"""
            
            elif any(word in message_lower for word in ['gracias', 'thank you', 'agradezco']):
                return "¡De nada! Es un placer ayudarte. Si necesitas más análisis o tienes otras preguntas, estaré aquí."
            
            elif any(word in message_lower for word in ['adiós', 'hasta luego', 'bye', 'chau']):
                return "¡Hasta luego! Ha sido genial poder ayudarte. Vuelve cuando necesites más análisis o asistencia con IA."
            
            elif '?' in message:
                # Respuesta para preguntas
                sentiment = self.analyze_sentiment(message)
                if sentiment['sentiment'] == 'NEGATIVE':
                    return f"Entiendo tu preocupación. Basándome en tu pregunta: '{message}', puedo decirte que estoy aquí para ayudarte a encontrar respuestas. ¿Podrías darme más contexto sobre lo que necesitas?"
                else:
                    return f"Interesante pregunta. Para poder ayudarte mejor con: '{message}', necesitaría un poco más de información o contexto. ¿Puedes darme más detalles?"
            
            else:
                # Análisis del sentimiento del mensaje para respuesta adaptativa
                sentiment_analysis = self.analyze_sentiment(message)
                keywords = self.extract_keywords(message, 5)
                
                response_parts = []
                
                if sentiment_analysis['sentiment'] == 'POSITIVE':
                    response_parts.append("Me alegra tu mensaje positivo.")
                elif sentiment_analysis['sentiment'] == 'NEGATIVE':
                    response_parts.append("Entiendo que puedas estar preocupado o frustrado.")
                
                if keywords:
                    response_parts.append(f"Veo que mencionas temas relacionados con: {', '.join(keywords[:3])}.")
                
                response_parts.append("Como asistente de IA, puedo ayudarte con análisis de texto, responder consultas y procesar información. ¿Hay algo específico en lo que pueda asistirte?")
                
                return " ".join(response_parts)
            
        except Exception as e:
            logger.error(f"❌ Error generando respuesta de chat: {e}")
            return "Disculpa, tuve un pequeño problema procesando tu mensaje. ¿Podrías reformularlo o intentar de nuevo?"
    
    # ===================================================================
    # CLASIFICACIÓN DE TEXTO
    # ===================================================================
    
    async def classify_text(self, request: ClassificationRequest) -> Dict[str, Any]:
        """Clasificar texto en categorías"""
        try:
            text = request.text
            categories = request.categories
            
            text_lower = text.lower()
            keywords_by_category = {}
            
            # Extraer palabras clave del texto
            text_keywords = set(self.extract_keywords(text, 20))
            
            # Calcular similitud con cada categoría
            category_scores = {}
            
            for category in categories:
                category_lower = category.lower()
                # Buscar menciones directas de la categoría
                direct_mentions = text_lower.count(category_lower)
                
                # Buscar palabras relacionadas
                related_words = 0
                if 'tecnología' in category_lower or 'tech' in category_lower:
                    tech_words = ['software', 'hardware', 'digital', 'internet', 'app', 'web', 'sistema', 'código']
                    related_words = sum(1 for word in tech_words if word in text_lower)
                
                elif 'negocios' in category_lower or 'business' in category_lower:
                    business_words = ['empresa', 'ventas', 'cliente', 'mercado', 'producto', 'servicio', 'negocio']
                    related_words = sum(1 for word in business_words if word in text_lower)
                
                elif 'salud' in category_lower or 'health' in category_lower:
                    health_words = ['médico', 'hospital', 'tratamiento', 'enfermedad', 'salud', 'medicina']
                    related_words = sum(1 for word in health_words if word in text_lower)
                
                score = direct_mentions * 2 + related_words
                category_scores[category] = score
            
            # Obtener la categoría con mayor score
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                confidence = min(0.95, 0.5 + (best_category[1] / max(len(text.split()), 1)))
                
                return {
                    "text": text,
                    "predicted_category": best_category[0],
                    "confidence": float(confidence),
                    "all_scores": {cat: float(score) for cat, score in category_scores.items()},
                    "method": "keyword_matching"
                }
            else:
                return {
                    "text": text,
                    "predicted_category": categories[0] if categories else "unknown",
                    "confidence": 0.5,
                    "method": "default"
                }
            
        except Exception as e:
            logger.error(f"❌ Error en clasificación: {e}")
            raise HTTPException(status_code=500, detail=f"Error en clasificación: {str(e)}")

# ===================================================================
# INSTANCIA GLOBAL DEL SERVICIO
# ===================================================================

ai_service = AIEngineService()

# ===================================================================
# ENDPOINTS DE LA API
# ===================================================================

@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "message": "🤖 Bienvenido al Motor de IA - Agente IA OyP 6.0",
        "service": "ai-engine",
        "version": "6.0.0",
        "status": "active",
        "capabilities": [
            "text_analysis",
            "sentiment_analysis",
            "entity_extraction", 
            "text_summarization",
            "keyword_extraction",
            "language_detection",
            "chat_conversation",
            "text_classification"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check del servicio"""
    return {
        "status": "healthy",
        "service": "ai-engine",
        "port": 8001,
        "timestamp": datetime.now().isoformat(),
        "conversations_active": len(ai_service.conversations)
    }

@app.get("/info")
async def service_info():
    """Información detallada del servicio"""
    return {
        "name": "ai-engine",
        "description": "Motor de IA con análisis de texto avanzado y chat conversacional",
        "port": 8001,
        "version": "6.0.0",
        "endpoints": {
            "GET /": "Información del servicio",
            "GET /health": "Health check",
            "GET /info": "Información detallada",
            "POST /analyze": "Análisis completo de texto",
            "POST /chat": "Chat conversacional",
            "POST /summarize": "Resumen de texto",
            "POST /classify": "Clasificación de texto",
            "GET /conversations/{id}": "Historial de conversación"
        }
    }

@app.post("/analyze")
async def analyze_text(request: TextAnalysisRequest):
    """Endpoint para análisis completo de texto"""
    return await ai_service.analyze_text_complete(request)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Endpoint para chat conversacional"""
    return await ai_service.process_chat(request)

@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    """Endpoint para resumen de texto"""
    summary_result = ai_service.generate_summary(request.text)
    return {
        "text": request.text,
        "summary": summary_result["summary"],
        "method": summary_result["method"],
        "compression_ratio": summary_result.get("compression_ratio", 0),
        "original_length": summary_result.get("original_length", 0),
        "summary_length": summary_result.get("summary_length", 0)
    }

@app.post("/classify")
async def classify_text(request: ClassificationRequest):
    """Endpoint para clasificación de texto"""
    return await ai_service.classify_text(request)

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Obtener historial de conversación"""
    if conversation_id in ai_service.conversations:
        return {
            "conversation_id": conversation_id,
            "messages": ai_service.conversations[conversation_id],
            "message_count": len(ai_service.conversations[conversation_id])
        }
    else:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

@app.get("/conversations")
async def list_conversations():
    """Listar todas las conversaciones activas"""
    return {
        "total_conversations": len(ai_service.conversations),
        "conversation_ids": list(ai_service.conversations.keys())
    }

# ===================================================================
# INICIALIZACIÓN DEL SERVICIO
# ===================================================================

if __name__ == "__main__":
    logger.info("🚀 Iniciando AI Engine Service...")
    logger.info("📍 Puerto: 8001")
    logger.info("📖 Documentación: http://localhost:8001/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )