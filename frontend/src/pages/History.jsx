import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts'

const API = 'http://localhost:8000'
const GOAL_ML = 2000

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#1e293b', border: '1px solid #334155',
      borderRadius: 10, padding: '10px 14px', fontSize: 13
    }}>
      <p style={{ color: '#94a3b8', marginBottom: 4 }}>{label}</p>
      <p style={{ color: '#f1f5f9', fontWeight: 700 }}>{payload[0].value} ml</p>
      <p style={{ color: '#94a3b8' }}>{Math.round(payload[0].value / 15)} sips</p>
    </div>
  )
}

export default function History() {
  const navigate = useNavigate()
  const [history, setHistory] = useState([])
  const [error, setError] = useState('')

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (!token) { navigate('/login'); return }

    fetch(`${API}/sips/history?days=30`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(res => {
        if (res.status === 401) { navigate('/login'); return }
        return res.json()
      })
      .then(data => {
        if (data) setHistory(data)
      })
      .catch(() => setError('Could not reach the server.'))
  }, [])

  function logout() {
    localStorage.clear()
    navigate('/login')
  }

  const totalMl = history.reduce((sum, d) => sum + d.total_ml, 0)
  const avgMl = history.length ? Math.round(totalMl / history.length) : 0
  const best = history.length ? Math.max(...history.map(d => d.total_ml)) : 0
  const daysHitGoal = history.filter(d => d.total_ml >= GOAL_ML).length

  // Format date labels: "Mon 3"
  const chartData = history.map(d => ({
    ...d,
    label: new Date(d.date + 'T00:00:00').toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric' })
  }))

  return (
    <div className="page">
      <nav>
        <div className="inner">
          <span className="logo">💧 Water Tracker</span>
          <div className="nav-links">
            <Link to="/dashboard">Dashboard</Link>
            <Link to="/history" className="active">History</Link>
          </div>
          <button className="logout" onClick={logout}>Log out</button>
        </div>
      </nav>

      <div className="container">
        <div className="page-header">
          <h1>Your History</h1>
          <p className="muted">Last 30 days</p>
        </div>

        {error && <div className="error-msg">{error}</div>}

        <div className="card-grid">
          <div className="stat-card">
            <div className="label">Daily average</div>
            <div className="value">{avgMl}<span className="unit">ml</span></div>
          </div>
          <div className="stat-card">
            <div className="label">Best day</div>
            <div className="value">{Math.round(best)}<span className="unit">ml</span></div>
          </div>
          <div className="stat-card">
            <div className="label">Goal hit</div>
            <div className="value">{daysHitGoal}<span className="unit">days</span></div>
          </div>
          <div className="stat-card">
            <div className="label">Total (30d)</div>
            <div className="value">{(totalMl / 1000).toFixed(1)}<span className="unit">L</span></div>
          </div>
        </div>

        <div className="card chart-section">
          <h2>Daily intake (ml)</h2>
          {chartData.length === 0 ? (
            <p className="muted" style={{ textAlign: 'center', padding: '40px 0' }}>
              No data yet. Start tracking to see your history here.
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData} barCategoryGap="30%">
                <XAxis
                  dataKey="label"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={v => `${v}ml`}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                <ReferenceLine
                  y={GOAL_ML}
                  stroke="#334155"
                  strokeDasharray="4 4"
                  label={{ value: 'Goal', fill: '#64748b', fontSize: 11, position: 'right' }}
                />
                <Bar dataKey="total_ml" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.total_ml >= GOAL_ML ? '#34d399' : '#38bdf8'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  )
}
