# Physical AI & Humanoid Robotics Textbook

An interactive, AI-powered textbook for learning Physical AI and Humanoid Robotics, built with Docusaurus and featuring a RAG-powered chatbot using Google Gemini.

## Features

- **Comprehensive Curriculum**: 4 modules covering ROS 2, simulation, NVIDIA Isaac, and Vision-Language-Action
- **RAG Chatbot**: AI assistant powered by Google Gemini for answering questions about the textbook content
- **Urdu Translation**: Built-in translation support for Pakistani students (50 bonus points feature)
- **Interactive Diagrams**: Mermaid.js diagrams for visualizing complex concepts
- **Responsive Design**: Mobile-friendly design for learning on any device

## Quick Start

### Frontend (Docusaurus)

```bash
cd docusaurus
npm install
npm start
```

Visit `http://localhost:3000` to view the textbook.

### Backend (RAG Chatbot)

```bash
# From project root
python run_backend.py

# Or from backend directory
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

### Health Check

Verify all services are properly configured:

```bash
cd backend
python health_check.py
```

## One-Click Deployment

### Frontend (Vercel)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/humanoid-robotics-course-book)

The frontend is automatically deployed to Vercel. Configuration is in `vercel.json`.

### Backend (Railway)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/YOUR_USERNAME/humanoid-robotics-course-book&plugins=postgresql&envs=GOOGLE_API_KEY,QDRANT_URL,QDRANT_API_KEY,CORS_ORIGINS&optionalEnvs=APP_ENV)

Required environment variables for Railway:
- `GOOGLE_API_KEY`: Your Google AI API key
- `DATABASE_URL`: PostgreSQL connection string (auto-provided by Railway)
- `QDRANT_URL`: Qdrant Cloud endpoint
- `QDRANT_API_KEY`: Qdrant API key
- `CORS_ORIGINS`: Allowed origins (e.g., `https://your-vercel-app.vercel.app`)

## Project Structure

```
humanoid-robotics-course-book/
├── docusaurus/              # Frontend (Docusaurus)
│   ├── docs/                # Markdown course content
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ChatWidget/  # RAG chatbot UI
│   │   │   └── UrduTranslation/  # Translation button
│   │   └── theme/           # Theme customizations
│   └── static/              # Static assets
├── backend/                 # Backend (FastAPI)
│   ├── app/
│   │   ├── api/             # API routes
│   │   ├── models/          # Database models
│   │   └── services/        # Business logic
│   ├── health_check.py      # Service health check
│   ├── railway.toml         # Railway deployment config
│   └── Procfile             # Railway/Heroku process file
├── run_backend.py           # Backend runner script
└── vercel.json              # Vercel deployment config
```

## Environment Variables

### Backend (.env)

```env
# Google Gemini
GOOGLE_API_KEY=your_google_api_key

# PostgreSQL (Neon)
DATABASE_URL=postgresql+asyncpg://user:pass@host/db?ssl=require

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Application
APP_ENV=development
CORS_ORIGINS=http://localhost:3000,https://your-production-url.com

# Model Configuration
EMBEDDING_MODEL=models/text-embedding-004
CHAT_MODEL=gemini-1.5-flash
```

## Course Modules

1. **Module 1: The Robotic Nervous System** - ROS 2 fundamentals
2. **Module 2: The Digital Twin** - Simulation with Gazebo and Unity
3. **Module 3: The AI-Robot Brain** - NVIDIA Isaac platform
4. **Module 4: Vision-Language-Action** - Embodied AI and VLA

## Technologies

- **Frontend**: Docusaurus 3.x, React 18, TypeScript
- **Backend**: FastAPI, Python 3.12
- **AI**: Google Gemini (text-embedding-004, gemini-1.5-flash)
- **Database**: PostgreSQL (Neon), Qdrant (vector DB)
- **Deployment**: Vercel (frontend), Railway (backend)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details.

---

Built for the Panaversity Q4 Hackathon
