from gateway.config import settings

def test_settings():
    print("Testing settings...")
    print(f"DEBUG: {settings.DEBUG}")
    print(f"ENVIRONMENT: {settings.ENVIRONMENT}")
    print(f"HOST: {settings.HOST}")
    print(f"PORT: {settings.PORT}")
    print(f"WORKERS: {settings.WORKERS}")
    print(f"CORS_ORIGINS: {settings.CORS_ORIGINS}")
    print(f"DATABASE_URL: {settings.DATABASE_URL}")
    print(f"SECRET_KEY: {'*' * len(settings.SECRET_KEY) if settings.SECRET_KEY else 'Not set'}")
    print(f"AI_ENGINE_URL: {settings.AI_ENGINE_URL}")
    print(f"DOC_PROCESSOR_URL: {settings.DOC_PROCESSOR_URL}")
    print(f"ANALYTICS_URL: {settings.ANALYTICS_URL}")
    print(f"REPORTS_URL: {settings.REPORTS_URL}")

if __name__ == "__main__":
    test_settings()
