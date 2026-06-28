# Migration Guide: ChromaDB to PostgreSQL + pgvector

This document explains how to migrate the Court Case Q&A backend to use PostgreSQL + pgvector, enable semantic section-role chunking, and connect to the external HTTP embedder.

---

## Prerequisites

1. **PostgreSQL 12+** must be installed and running on your system.
2. **pgvector extension** must be installed on your PostgreSQL server.
   - For Windows, you can find installation instructions on the [pgvector GitHub repository](https://github.com/pgvector/pgvector).

---

## Migration Steps

### Step 1: Create the Database & Initialize Schema
1. Connect to your PostgreSQL server using `psql`, `pgAdmin`, or any SQL client:
   ```sql
   CREATE DATABASE court_cases;
   ```
2. Run the SQL statements inside [schema.sql](file:///C:/llm%20model/schema.sql) against the `court_cases` database. Note that the schema creation will automatically create the `vector` extension and define the tables and indices.
   *(Alternatively, the FastAPI application will automatically check for the tables on startup and execute `schema.sql` if they do not exist).*

### Step 2: Configure Environment Variables
Open or create the [.env](file:///C:/llm%20model/.env) file in the project root and add/configure the following keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/court_cases
EMBEDDER_URL=http://localhost:8000/embed
EMBEDDER_DIM=2560
```
Make sure to adjust the database username (`postgres`), password (`password`), host, and port as needed.

### Step 3: Install Dependencies
Run the following command inside your virtual environment to install the updated Python packages:
```bash
.\venv\Scripts\pip install -r requirements.txt
```

### Step 4: Run the Application
Start the FastAPI server:
```bash
.\venv\Scripts\python.exe main.py
```
On startup:
- The server will execute the `schema.sql` template, replacing `{{INSERT_DIM}}` with the `EMBEDDER_DIM` environment variable.
- It will automatically scan the folder `./court_cases_db/8e6371b3-4bc5-4b8d-9f8c-5778a1986bd0` for PDF files.
- It will parse and chunk any newly found PDFs semantically using Gemini, call the embedder, and save them in PostgreSQL.

---

## Verification
You can verify the backend is running and the cases are successfully loaded by calling these endpoints:
1. **Health Check**:
   - URL: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
   - Expected Output: `{"status": "ok"}`
2. **Cases List**:
   - URL: [http://127.0.0.1:8000/cases](http://127.0.0.1:8000/cases)
   - Expected Output: `[ { "case_name": "filename.pdf", "page_count": X } ]`
