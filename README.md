# FinTech Compliance Document Simplifier

Easily simplify complex financial compliance documents using AI! This application parses regulatory and financial documents (PDF, DOCX, TXT) and translates complex legal jargon into plain, actionable English using NVIDIA LLM endpoints via LangChain.

The frontend is styled in a premium, developer-friendly **Dracula theme** (using official Dracula color tokens) and features a responsive two-column workspace.

---

## 🛠️ Prerequisites

Before running the application, make sure you have the following installed:
1. **Docker & Docker Compose** (v2.0+)
2. **NVIDIA API Key** (Required for ChatNVIDIA endpoint integration)

---

## ⚙️ Configuration

1. Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your details:
   ```env
   NVIDIA_API_KEY=your_nvidia_api_key_here
   NVIDIA_MODEL_NAME=minimaxai/minimax-m3
   ```

---

## 🚀 Deployment Guidelines (Docker Compose)

You can launch both the frontend and backend services concurrently using Docker Compose profiles.

### 1. Run in Development Mode
Builds development images and starts servers with hot-reloading active. Changes in source directories reflect instantly:
```bash
docker compose --profile dev up --build
```
* **Frontend UI**: [http://localhost:5173](http://localhost:5173)
* **Backend API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Run in Production Mode
Builds optimized production stages. Static assets are built and served by a high-performance **Nginx** server, which proxies api queries internally:
```bash
docker compose --profile prod up --build
```
* **Application URL**: [http://localhost](http://localhost) (Exposed directly on port 80)
* **Internal Backend Swagger**: [http://localhost:8080/docs](http://localhost:8080/docs)

---

## 🧹 Destroy Guidelines (Cleanup)

To stop services, release bound network ports, and cleanly destroy active containers and networks:

### 1. Stop Development Services
```bash
docker compose --profile dev down
```

### 2. Stop Production Services
```bash
docker compose --profile prod down
```

### 3. Deep Clean (Remove volumes and orphan containers)
```bash
docker compose --profile dev down -v --remove-orphans
# or
docker compose --profile prod down -v --remove-orphans
```

---

## 💻 Local Development (Without Docker)

### Backend
1. Navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Install dependencies and start the uvicorn dev server:
   ```bash
   uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

### Frontend
1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install packages and start the Vite dev server:
   ```bash
   npm install
   npm run dev
   ```
