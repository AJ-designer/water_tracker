import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'

const API = 'http://localhost:8000'
const GOAL_ML = 2000

function ProgressRing({ percent, totalMl }) {
  const radius = 80
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (percent / 100) * circumference

  const color = percent >= 100 ? '#34d399' : percent >= 50 ? '#38bdf8' : '#f87171'

  return (
    <div className="ring-wrapper">
      <div className="ring-container">
        <svg width="200" height="200" viewBox="0 0 200 200">
          {/* Track */}
          <circle cx="100" cy="100" r={radius} fill="none" stroke="#293548" strokeWidth="16" />
          {/* Progress */}
          <circle
            cx="100" cy="100" r={radius}
            fill="none"
            stroke={color}
            strokeWidth="16"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 0.6s ease, stroke 0.4s ease' }}
          />
        </svg>
        <div className="ring-center">
          <span className="ring-ml">{Math.round(totalMl)}</span>
          <span className="ring-label">ml today</span>
        </div>
      </div>
      <span className="ring-goal">Goal: {GOAL_ML}ml — {percent.toFixed(0)}% complete</span>
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const email = localStorage.getItem('email') || ''
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')

  async function fetchStats() {
    const token = localStorage.getItem('token')
    if (!token) { navigate('/login'); return }

    try {
      const res = await fetch(`${API}/sips/stats`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (res.status === 401) { navigate('/login'); return }
      const data = await res.json()
      setStats(data)
    } catch {
      setError('Could not reach the server.')
    }
  }

  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, 5000)
    return () => clearInterval(interval)
  }, [])

  function logout() {
    localStorage.clear()
    navigate('/login')
  }

  return (
    <div className="page">
      <nav>
        <div className="inner">
          <span className="logo">💧 Water Tracker</span>
          <div className="nav-links">
            <Link to="/dashboard" className="active">Dashboard</Link>
            <Link to="/history">History</Link>
          </div>
          <button className="logout" onClick={logout}>Log out</button>
        </div>
      </nav>

      <div className="container">
        <div className="page-header">
          <h1>Today's Progress</h1>
          <p className="muted">{email}</p>
        </div>

        {error && <div className="error-msg">{error}</div>}

        {stats && (
          <>
            <div className="card">
              <ProgressRing percent={stats.goal_percent} totalMl={stats.today_ml} />
            </div>

            <div className="card-grid">
              <div className="stat-card">
                <div className="label">Sips today</div>
                <div className="value">{stats.today_sips}</div>
              </div>
              <div className="stat-card">
                <div className="label">Weekly avg</div>
                <div className="value">
                  {Math.round(stats.weekly_average_ml)}
                  <span className="unit">ml/day</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="label">All-time total</div>
                <div className="value">
                  {(stats.all_time_ml / 1000).toFixed(1)}
                  <span className="unit">L</span>
                </div>
              </div>
              <div className="stat-card">
                <div className="label">All-time sips</div>
                <div className="value">{stats.all_time_sips}</div>
              </div>
            </div>

            <p className="refresh-note muted">Refreshes every 5 seconds while the tracker is running</p>
          </>
        )}
      </div>
    </div>
  )
}
