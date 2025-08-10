# Dockerfile for the FinTech Compliance Document Simplifier application

# Base stage with common dependencies
FROM python:3.13-slim AS base

# Set working directory
WORKDIR /Compliance_Document_Simplifier

# Copy requirements file
COPY requirements.txt .

# Update pip to the latest version
RUN pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base AS development

# Set environment variable
ENV environment=development

# In development, we'll mount the app directory as a volume
# so no need to copy files here
EXPOSE 8000

# Command for development (with hot reload)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production

# Set environment variable
ENV environment=production

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Command for production
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
