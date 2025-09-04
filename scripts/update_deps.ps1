# Script to update Python dependencies with specific versions

Write-Host "Updating Python dependencies..." -ForegroundColor Cyan

# Create a backup of current dependencies
pip freeze > requirements_backup.txt
Write-Host "✅ Created backup of current dependencies in requirements_backup.txt" -ForegroundColor Green

# Install specific versions
pip install -r requirements-updated.txt --upgrade

Write-Host "✅ Dependencies updated successfully!" -ForegroundColor Green
Write-Host "You can now start the server with: python -m uvicorn gateway.main:app --reload" -ForegroundColor Yellow
