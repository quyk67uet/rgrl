# VHAS Demo - Vietnam Health-Agent System

A Proof-of-Concept (POC) demonstration of the VHAS framework, showcasing a clinical workflow orchestrator with real-time visualization.

## 🏗️ Architecture

### Frontend (Next.js 15)
- **3-Column Layout**: Control Panel, Chat Interface, and Live Workflow Visualizer
- **Real-time Updates**: Server-Sent Events (SSE) for streaming workflow execution
- **Modern UI**: Built with React, TypeScript, and Tailwind CSS

### Backend (FastAPI)
- **Clinical Workflow Orchestrator**: Coordinates specialized agents
- **Worker Agents**: Triage, EHR, Pharmacy, and Summary agents
- **Tool System**: Mocked clinical tools (EWS scoring, vitals, medications, etc.)
- **SSE Streaming**: Real-time workflow trace broadcasting

## 📁 Project Structure

```
vhas-demo/
├── app/
│   └── page.tsx              # Main UI with 3-column layout
├── components/
│   ├── ChatInterface.tsx     # Chat history component
│   └── WorkflowVisualizer.tsx # Live workflow trace visualizer
├── backend/
│   ├── main.py               # FastAPI server & Orchestrator
│   ├── agents.py             # Worker Agent classes
│   ├── tools.py              # Mocked clinical tools
│   ├── vhas_universe.json    # Agent/tool definitions
│   └── requirements.txt      # Python dependencies
├── package.json              # Node.js dependencies
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- **pip** (Python package manager)

### Installation

#### 1. Install Frontend Dependencies

```bash
npm install
```

#### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### Running the Application

You'll need to run both the backend and frontend servers simultaneously.

#### Terminal 1: Start the Backend (FastAPI)

```bash
cd backend
python main.py
```

The backend will start on `http://localhost:8000`

You can verify it's running by visiting:
- `http://localhost:8000` - API root
- `http://localhost:8000/api/health` - Health check endpoint

#### Terminal 2: Start the Frontend (Next.js)

```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

### Using the Demo

1. Open your browser to `http://localhost:3000`
2. In the **Control Panel** (left column):
   - Enter a Patient ID (default: `P12345`)
   - Enter symptoms (default: `chest pain and difficulty breathing`)
   - Click **"Start Clinical Workflow"**
3. Watch the workflow execute in real-time:
   - **Chat Interface** (middle): Shows user commands and final AI summaries
   - **Workflow Visualizer** (right): Displays live agent/tool execution traces

## 🔄 Workflow Sequence

The hard-coded workflow follows this "Standard Loop" pathway:

1. **TriageAgent** → Calculates Early Warning Score (EWS)
2. **EHRAgent** → Records vital signs
3. **PharmacyAgent** → Retrieves medication history
4. **EHRAgent** → Adds clinical note
5. **PharmacyAgent** → Checks drug interactions
6. **SummaryAgent** → Generates final clinical summary

## 🎨 Visual Indicators

### Workflow Visualizer Color Coding:
- **Purple** 🟣 - Orchestrator Decisions
- **Blue** 🔵 - Agent Executions
- **Green** 🟢 - Tool Calls

## 🛠️ Technical Details

### Backend Technologies
- **FastAPI**: Modern Python web framework
- **SSE (Server-Sent Events)**: Real-time streaming
- **sse-starlette**: SSE support for FastAPI
- **Uvicorn**: ASGI server

### Frontend Technologies
- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **EventSource API**: Native SSE client

### Key Design Patterns
- **Agentic Design**: Specialized, single-responsibility agents
- **Orchestrator Pattern**: Central coordinator for workflow routing
- **Observable Workflow**: Real-time trace visualization for interpretability

## 📊 API Endpoints

### `GET /api/workflow`
Streams workflow execution via SSE.

**Query Parameters:**
- `patient_id` (string): Patient identifier
- `symptoms` (string): Patient symptoms description

**Response:** Server-Sent Events stream with workflow spans

### `GET /api/health`
Health check endpoint.

**Response:** JSON with service status

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the root directory (optional):

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NODE_ENV=development
```

### Backend Configuration

The backend runs on port 8000 by default. To change this, modify `backend/main.py`:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=YOUR_PORT)
```

## 🧪 Testing the System

### Test Different Scenarios

Try different symptoms to see how the workflow responds:

1. **High Priority Case:**
   - Symptoms: `chest pain and difficulty breathing`
   - Expected: EWS score of 5 (High)

2. **Medium Priority Case:**
   - Symptoms: `mild headache`
   - Expected: EWS score of 2 (Medium)

3. **Drug Interaction Case:**
   - Patient ID: `P12345` (has Lisinopril in history)
   - The workflow will detect interaction with Ibuprofen

## 🐛 Troubleshooting

### Backend Not Starting
- Ensure all Python dependencies are installed: `pip install -r backend/requirements.txt`
- Check if port 8000 is already in use
- Verify Python version: `python --version` (should be 3.8+)

### Frontend Not Connecting to Backend
- Verify backend is running on `http://localhost:8000`
- Check browser console for CORS errors
- Ensure both servers are running simultaneously

### SSE Connection Issues
- Check browser compatibility (all modern browsers support EventSource)
- Verify network tab shows connection to `/api/workflow`
- Check backend logs for errors

## 🚧 Future Enhancements

This is a POC with hard-coded workflows. Future versions could include:

- **AI/RL Integration**: Dynamic workflow routing using reinforcement learning
- **Real EHR Integration**: Connect to actual Electronic Health Record systems
- **Multi-Agent Coordination**: Parallel agent execution
- **Workflow Templates**: Multiple pre-defined clinical pathways
- **Authentication**: User login and role-based access control
- **Persistence**: Database storage for workflow history
- **Analytics Dashboard**: Workflow performance metrics

## 📝 License

This is a demonstration project for educational purposes.

## 🤝 Contributing

This is a POC demo. For production use, consider:
- Adding comprehensive error handling
- Implementing authentication and authorization
- Adding database persistence
- Writing unit and integration tests
- Implementing logging and monitoring
- Adding input validation and sanitization

## 📧 Support

For questions or issues, please refer to the VHAS framework documentation.

---

**Built with ❤️ using Next.js, FastAPI, and modern web technologies**
