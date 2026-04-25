import { useState, useEffect } from 'react'
import './App.css'

const API_BASE = "http://localhost:8000"

function App() {
  const [isRunning, setIsRunning] = useState(false)
  const [status, setStatus] = useState<boolean>(false)
  const [loading, setLoading] = useState(false)

  // Polling the /status endpoint when running
  useEffect(() => {
    let interval: number | undefined;

    if (isRunning) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/status`)
          const data = await res.json()
          setStatus(data.person_present)
          setIsRunning(data.is_running)
        } catch (error) {
          console.error("Error fetching status:", error)
        }
      }, 500) // Poll every 500ms
    } else {
      setStatus(false)
      if (interval) clearInterval(interval)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isRunning])

  const toggleDetection = async () => {
    setLoading(true)
    try {
      if (isRunning) {
        await fetch(`${API_BASE}/stop`)
        setIsRunning(false)
      } else {
        await fetch(`${API_BASE}/start`)
        setIsRunning(true)
      }
    } catch (error) {
      console.error("Error toggling detection:", error)
      alert("Backend not reachable. Make sure FastAPI is running on port 8000.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1>Lightweight Real-Time Person Detection</h1>
        <p>Optimized for Edge Devices & IoT Integrations</p>
      </header>

      <main className="dashboard">
        <div className="controls">
          <button 
            className={`btn ${isRunning ? 'btn-stop' : 'btn-start'}`}
            onClick={toggleDetection}
            disabled={loading}
          >
            {loading ? "Please wait..." : isRunning ? "Stop Detection" : "Start Camera"}
          </button>
        </div>

        <div className="status-panel">
          <h2>Stable Binary Signal:</h2>
          <div className={`signal-indicator ${status ? 'signal-true' : 'signal-false'}`}>
            {status ? "PERSON PRESENT (TRUE)" : "NO PERSON (FALSE)"}
          </div>
          <p className="explanation">
            <em>Output is temporally smoothed. Signal only changes to FALSE after 2 seconds of continuous absence to prevent flickering.</em>
          </p>
        </div>

        <div className="video-panel">
          <h2>Live Feed</h2>
          <div className="video-container">
            {isRunning ? (
              <img 
                src={`${API_BASE}/video_feed`} 
                alt="Live Video Feed" 
                className="video-stream"
              />
            ) : (
              <div className="video-placeholder">
                <p>Camera is currently off.</p>
                <p>Click "Start Camera" to begin.</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
