import { useEffect, useMemo, useState } from "react";
import { getHerdRisk, getPriorityCows } from "./services/api";
import Login from "./components/Login";

import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  BarChart3,
  Bell,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Clock3,
  Cloud,
  Database,
  Droplets,
  Eye,
  Gauge,
  HeartPulse,
  Leaf,
  LogOut,
  RefreshCw,
  Search,
  ShieldCheck,
  Thermometer,
  TrendingUp,
  Users,
  Wifi,
  XCircle,
} from "lucide-react";

import "./App.css";


function App() {

  /* ============================================================
     AUTHENTICATION
  ============================================================ */

  const [isAuthenticated, setIsAuthenticated] = useState(
    () => Boolean(localStorage.getItem("token"))
  );

  const [username, setUsername] = useState(
    () => localStorage.getItem("username") || ""
  );

  const [herdData, setHerdData] = useState(null);
  const [priorityData, setPriorityData] = useState(null);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const [error, setError] = useState("");

  const [searchTerm, setSearchTerm] = useState("");
  const [riskFilter, setRiskFilter] = useState("ALL");

  const [lastUpdated, setLastUpdated] = useState(null);


  /* ============================================================
     LOGIN SUCCESS
  ============================================================ */

  const handleLogin = (data) => {

    setIsAuthenticated(true);

    setUsername(
      data.username ||
      localStorage.getItem("username") ||
      ""
    );
  };


  /* ============================================================
     LOGOUT
  ============================================================ */

  const handleLogout = () => {

    localStorage.removeItem("token");
    localStorage.removeItem("username");
    localStorage.removeItem("role");

    setIsAuthenticated(false);
    setUsername("");

    setHerdData(null);
    setPriorityData(null);
    setError("");
  };


  /* ============================================================
     DASHBOARD API
  ============================================================ */

  const loadDashboard = async (isRefresh = false) => {

    try {

      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError("");

      const [herdResult, priorityResult] = await Promise.all([
        getHerdRisk(),
        getPriorityCows(),
      ]);

      setHerdData(herdResult);
      setPriorityData(priorityResult);

      setLastUpdated(new Date());

    } catch (err) {

      console.error("Dashboard loading error:", err);

      /*
       * If JWT is invalid or expired,
       * send the user back to login.
       */

      if (
        err.message?.includes("401") ||
        err.message?.toLowerCase().includes("unauthorized")
      ) {

        handleLogout();

        return;
      }

      setError(
        "Unable to connect to the Mastitis AI backend. Make sure the FastAPI server is running on port 8000."
      );

    } finally {

      setLoading(false);
      setRefreshing(false);
    }
  };


  /* ============================================================
     LOAD DASHBOARD ONLY AFTER LOGIN
  ============================================================ */

  useEffect(() => {

    if (!isAuthenticated) {
      return;
    }

    loadDashboard();

  }, [isAuthenticated]);


  /* ============================================================
     FILTER PRIORITY COWS
     
     IMPORTANT:
     useMemo MUST be called before the authentication return.
     Otherwise React detects a change in Hook order.
  ============================================================ */

  const priorityCows = useMemo(() => {

    if (!priorityData?.priority_cows) {
      return [];
    }

    let cows = [...priorityData.priority_cows];

    if (searchTerm.trim()) {

      const search = searchTerm.toLowerCase();

      cows = cows.filter((cow) =>
        String(cow.cow_id || "")
          .toLowerCase()
          .includes(search)
      );
    }

    if (riskFilter !== "ALL") {

      cows = cows.filter(
        (cow) =>
          String(cow.risk_category || "").toUpperCase() === riskFilter
      );
    }

    return cows;

  }, [priorityData, searchTerm, riskFilter]);


  /* ============================================================
     SHOW LOGIN PAGE
     
     This MUST come AFTER all Hooks.
  ============================================================ */

  if (!isAuthenticated) {

    return (
      <Login onLogin={handleLogin} />
    );
  }


  /* ============================================================
     RISK HELPERS
  ============================================================ */

  const getRiskClass = (category = "") => {

    const value = category.toLowerCase();

    if (value.includes("high")) {
      return "risk-high";
    }

    if (value.includes("moderate")) {
      return "risk-moderate";
    }

    if (value.includes("low")) {
      return "risk-low";
    }

    return "risk-safe";
  };


  const getRiskIcon = (category = "") => {

    const value = category.toLowerCase();

    if (value.includes("high")) {
      return <AlertTriangle size={15} />;
    }

    if (value.includes("moderate")) {
      return <CircleAlert size={15} />;
    }

    if (value.includes("low")) {
      return <Activity size={15} />;
    }

    return <CheckCircle2 size={15} />;
  };


  const getTrend = (cow) => {

    const current = Number(cow.current_risk || 0);
    const forecast7 = Number(cow["7d_probability"] || 0);

    if (forecast7 > current + 5) {

      return {
        label: "Increasing",
        className: "trend-up",
        icon: <ArrowUp size={14} />,
      };
    }

    if (forecast7 < current - 5) {

      return {
        label: "Decreasing",
        className: "trend-down",
        icon: <ArrowDown size={14} />,
      };
    }

    return {
      label: "Stable",
      className: "trend-stable",
      icon: <Activity size={14} />,
    };
  };


  const formatPercentage = (value) => {

    if (
      value === null ||
      value === undefined ||
      Number.isNaN(Number(value))
    ) {
      return "--";
    }

    return `${Number(value).toFixed(1)}%`;
  };


  const formatNumber = (value) => {

    if (
      value === null ||
      value === undefined ||
      Number.isNaN(Number(value))
    ) {
      return "--";
    }

    return Number(value).toFixed(2);
  };


  const getHerdStatusClass = (status = "") => {

    const value = status.toLowerCase();

    if (value === "high") {
      return "status-high";
    }

    if (value === "moderate") {
      return "status-moderate";
    }

    if (value === "low") {
      return "status-low";
    }

    return "status-normal";
  };


  const getStatusDescription = (status = "") => {

    const value = status.toLowerCase();

    if (value === "high") {
      return "Multiple herd-level risk indicators require attention.";
    }

    if (value === "moderate") {
      return "Some herd-level indicators require increased monitoring.";
    }

    if (value === "low") {
      return "Risk indicators are present but remain limited.";
    }

    return "No significant herd-level risk indicators detected.";
  };


  const getRecommendationIcon = (recommendation = "") => {

    const value = recommendation.toLowerCase();

    if (
      value.includes("high-risk") ||
      value.includes("priority")
    ) {
      return <AlertTriangle size={18} />;
    }

    if (value.includes("monitor")) {
      return <Eye size={18} />;
    }

    if (value.includes("review")) {
      return <BarChart3 size={18} />;
    }

    return <ShieldCheck size={18} />;
  };


  /* ============================================================
     LOADING
  ============================================================ */

  if (loading) {

    return (
      <div className="app-loading">

        <div className="loading-card">

          <div className="loading-logo">
            <HeartPulse size={28} />
          </div>

          <h1>Mastitis AI</h1>

          <p>Loading herd intelligence...</p>

          <div className="loading-spinner">
            <RefreshCw size={20} />
          </div>

        </div>

      </div>
    );
  }


  /* ============================================================
     ERROR
  ============================================================ */

  if (error && !herdData) {

    return (
      <div className="app-error">

        <div className="error-card">

          <div className="error-icon">
            <XCircle size={30} />
          </div>

          <h1>Backend connection failed</h1>

          <p>{error}</p>

          <button
            className="primary-button"
            onClick={() => loadDashboard()}
          >
            <RefreshCw size={17} />
            Try again
          </button>

          <div className="connection-help">

            <strong>Backend command:</strong>

            <code>
              python -m uvicorn main:app --reload
            </code>

          </div>

        </div>

      </div>
    );
  }


  const status = herdData?.herd_status || "UNKNOWN";


  /* ============================================================
     DASHBOARD
  ============================================================ */

  return (

    <div className="app">

      {/* =========================
          SIDEBAR
      ========================== */}

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">
            <HeartPulse size={23} />
          </div>

          <div>

            <div className="brand-name">
              Mastitis AI
            </div>

            <div className="brand-subtitle">
              Dairy Intelligence
            </div>

          </div>

        </div>


        <nav className="sidebar-nav">

          <div className="nav-section-title">
            MONITORING
          </div>

          <button className="nav-item active">
            <Gauge size={18} />
            <span>Overview</span>
          </button>


          <button className="nav-item">

            <Users size={18} />

            <span>
              Priority Cows
            </span>

            <span className="nav-count">
              {herdData?.high_risk_count ?? 0}
            </span>

          </button>


          <button className="nav-item">

            <Bell size={18} />

            <span>
              Alerts
            </span>

          </button>


          <div className="nav-section-title">
            SYSTEM
          </div>


          <button className="nav-item">

            <Wifi size={18} />

            <span>
              Sensor Monitor
            </span>

          </button>


          <button className="nav-item">

            <Database size={18} />

            <span>
              Data & Models
            </span>

          </button>

        </nav>


        {/* =========================
            SIDEBAR BOTTOM
        ========================== */}

        <div className="sidebar-bottom">

          <div className="prototype-badge">

            <div className="prototype-dot"></div>

            <div>

              <strong>
                Prototype Mode
              </strong>

              <span>
                Model outputs connected
              </span>

            </div>

          </div>


          {/* USER INFORMATION */}

          <div className="sidebar-user">

            <div className="sidebar-user-info">

              <div className="sidebar-user-icon">
                <UserIcon />
              </div>

              <div>

                <strong>
                  {username || "User"}
                </strong>

                <span>
                  {localStorage.getItem("role") || "farmer"}
                </span>

              </div>

            </div>


            <button
              className="logout-button"
              onClick={handleLogout}
              title="Logout"
            >
              <LogOut size={17} />
              Logout
            </button>

          </div>


          <div className="sidebar-footer">
            Monitor → Detect → Forecast
          </div>

        </div>

      </aside>


      {/* =========================
          MAIN CONTENT
      ========================== */}

      <main className="main-content">

        {/* HEADER */}

        <header className="topbar">

          <div>

            <div className="breadcrumb">
              Dairy Intelligence / Overview
            </div>

            <h1>
              Herd Overview
            </h1>

            <p className="page-description">
              AI-powered early risk monitoring and forecasting for bovine
              mastitis.
            </p>

          </div>


          <div className="topbar-actions">

            <div className="system-status">

              <span className="online-dot"></span>

              AI system online

            </div>


            <button
              className="refresh-button"
              onClick={() => loadDashboard(true)}
              disabled={refreshing}
              title="Refresh dashboard"
            >

              <RefreshCw
                size={17}
                className={refreshing ? "spin" : ""}
              />

              Refresh

            </button>

          </div>

        </header>


        {/* ERROR BANNER */}

        {error && (

          <div className="warning-banner">

            <AlertTriangle size={18} />

            <div>

              <strong>
                Connection warning
              </strong>

              <span>
                {error}
              </span>

            </div>

          </div>

        )}


        {/* =========================
            CURRENT HERD STATUS
        ========================== */}

        <section className="section">

          <div className="section-heading">

            <div>

              <span className="section-kicker">
                CURRENT HERD STATUS
              </span>

              <h2>
                How is the herd right now?
              </h2>

            </div>


            <div className="last-updated">

              <Clock3 size={15} />

              Last updated{" "}

              {lastUpdated
                ? lastUpdated.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : "--"}

            </div>

          </div>


          <div className="hero-status-grid">

            <div
              className={`herd-status-card ${getHerdStatusClass(status)}`}
            >

              <div className="status-card-top">

                <span>
                  HERD STATUS
                </span>

                <div className="status-icon">

                  {status === "HIGH" ? (
                    <AlertTriangle size={23} />
                  ) : status === "MODERATE" ? (
                    <CircleAlert size={23} />
                  ) : (
                    <ShieldCheck size={23} />
                  )}

                </div>

              </div>


              <div className="status-value">
                {status}
              </div>

              <p>
                {getStatusDescription(status)}
              </p>

            </div>


            <div className="metric-card">

              <div className="metric-icon">
                <Users size={21} />
              </div>

              <div className="metric-label">
                TOTAL COWS
              </div>

              <div className="metric-value">
                {herdData?.total_cows ?? "--"}
              </div>

              <div className="metric-caption">
                Animals monitored
              </div>

            </div>


            <div className="metric-card">

              <div className="metric-icon">
                <Activity size={21} />
              </div>

              <div className="metric-label">
                AVERAGE CURRENT RISK
              </div>

              <div className="metric-value">
                {formatPercentage(herdData?.average_risk)}
              </div>

              <div className="metric-caption">
                Model 1 herd average
              </div>

            </div>


            <div className="metric-card">

              <div className="metric-icon">
                <AlertTriangle size={21} />
              </div>

              <div className="metric-label">
                HIGH-RISK COWS
              </div>

              <div className="metric-value">
                {herdData?.high_risk_count ?? "--"}
              </div>

              <div className="metric-caption">
                Current risk ≥ high-risk threshold
              </div>

            </div>

          </div>

        </section>


        {/* =========================
            CURRENT RISK DISTRIBUTION
        ========================== */}

        <section className="section">

          <div className="section-heading">

            <div>

              <span className="section-kicker">
                MODEL 1
              </span>

              <h2>
                Current Herd Risk Distribution
              </h2>

            </div>

            <span className="section-note">
              Individual current-risk predictions
            </span>

          </div>


          <div className="distribution-card">

            <div className="distribution-bars">

              <RiskDistribution
                label="No Risk"
                count={herdData?.no_risk_count}
                total={herdData?.total_cows}
                className="distribution-safe"
              />

              <RiskDistribution
                label="Low Risk"
                count={herdData?.low_risk_count}
                total={herdData?.total_cows}
                className="distribution-low"
              />

              <RiskDistribution
                label="Moderate Risk"
                count={herdData?.moderate_risk_count}
                total={herdData?.total_cows}
                className="distribution-moderate"
              />

              <RiskDistribution
                label="High Risk"
                count={herdData?.high_risk_count}
                total={herdData?.total_cows}
                className="distribution-high"
              />

            </div>


            <div className="distribution-footer">

              <div>

                <strong>
                  Maximum current risk:
                </strong>{" "}

                {formatPercentage(
                  herdData?.maximum_risk
                )}

              </div>


              <div>

                <strong>
                  Median current risk:
                </strong>{" "}

                {formatPercentage(
                  herdData?.median_risk
                )}

              </div>

            </div>

          </div>

        </section>


        {/* =========================
            HERD RISK FORECAST
        ========================== */}

        <section className="section">

          <div className="section-heading">

            <div>

              <span className="section-kicker">
                MODEL 2
              </span>

              <h2>
                Herd Risk Forecast
              </h2>

              <p className="section-description">
                Future herd outlook aggregated from individual cow
                forecasting results.
              </p>

            </div>


            <div className="forecast-badge">

              <TrendingUp size={16} />

              AI Forecast

            </div>

          </div>


          <div className="forecast-grid">

            <ForecastCard
              title="7-Day Warning"
              value={formatPercentage(
                herdData?.warning_7d_percentage
              )}
              subtitle={`${herdData?.warning_7d_count ?? "--"} cows with elevated 7-day probability`}
              probability={
                herdData?.average_7d_probability
              }
            />


            <ForecastCard
              title="14-Day Warning"
              value={formatPercentage(
                herdData?.warning_14d_percentage
              )}
              subtitle={`${herdData?.warning_14d_count ?? "--"} cows with elevated 14-day probability`}
              probability={
                herdData?.average_14d_probability
              }
            />


            <ForecastCard
              title="Average 7-Day Probability"
              value={formatPercentage(
                herdData?.average_7d_probability
              )}
              subtitle="Average forecast probability across the herd"
              probability={
                herdData?.average_7d_probability
              }
            />


            <ForecastCard
              title="Average 14-Day Probability"
              value={formatPercentage(
                herdData?.average_14d_probability
              )}
              subtitle="Average longer-term forecast probability"
              probability={
                herdData?.average_14d_probability
              }
            />

          </div>


          <div className="forecast-explanation">

            <div className="explanation-icon">
              <TrendingUp size={20} />
            </div>

            <div>

              <strong>
                How to interpret this forecast
              </strong>

              <p>
                Model 1 estimates a cow's current mastitis risk.
                Model 2 analyzes the cow's historical risk trajectory
                to estimate future probability. Herd forecast values
                summarize those individual forecasts.
              </p>

            </div>

          </div>

        </section>


        {/* =========================
            PRIORITY COWS
        ========================== */}

        <section className="section">

          <div className="section-heading">

            <div>

              <span className="section-kicker">
                PRIORITIZATION
              </span>

              <h2>
                Priority Cows
              </h2>

              <p className="section-description">
                Animals requiring the closest attention based on current
                risk and future probability.
              </p>

            </div>


            <div className="priority-summary">

              <AlertTriangle size={16} />

              {priorityData?.priority_cows?.length ?? 0} priority
              animals

            </div>

          </div>


          <div className="table-toolbar">

            <div className="search-box">

              <Search size={17} />

              <input
                type="text"
                placeholder="Search by Cow ID..."
                value={searchTerm}
                onChange={(event) =>
                  setSearchTerm(event.target.value)
                }
              />

            </div>


            <div className="filter-group">

              {[
                "ALL",
                "HIGH RISK",
                "MODERATE RISK",
                "LOW RISK",
              ].map((filter) => (

                <button
                  key={filter}
                  className={
                    riskFilter === filter
                      ? "filter-button active"
                      : "filter-button"
                  }
                  onClick={() =>
                    setRiskFilter(filter)
                  }
                >

                  {filter === "ALL"
                    ? "All"
                    : filter
                        .replace(" RISK", "")
                        .toLowerCase()
                        .replace(/^./, (char) =>
                          char.toUpperCase()
                        )}

                </button>

              ))}

            </div>

          </div>


          <div className="table-card">

            {priorityCows.length === 0 ? (

              <div className="empty-state">

                <Search size={25} />

                <strong>
                  No matching cows
                </strong>

                <span>
                  Try changing the search or risk filter.
                </span>

              </div>

            ) : (

              <div className="table-wrapper">

                <table>

                  <thead>

                    <tr>
                      <th>COW</th>
                      <th>CURRENT RISK</th>
                      <th>CATEGORY</th>
                      <th>TREND</th>
                      <th>7-DAY FORECAST</th>
                      <th>14-DAY FORECAST</th>
                      <th>WARNING</th>
                      <th>ACTION</th>
                    </tr>

                  </thead>


                  <tbody>

                    {priorityCows.map((cow) => {

                      const trend = getTrend(cow);

                      return (

                        <tr key={cow.cow_id}>

                          <td>

                            <div className="cow-cell">

                              <div className="cow-avatar">
                                <HeartPulse size={16} />
                              </div>

                              <div>

                                <strong>
                                  {cow.cow_id}
                                </strong>

                                <span>
                                  Individual AI profile
                                </span>

                              </div>

                            </div>

                          </td>


                          <td>

                            <div className="risk-number">
                              {formatPercentage(
                                cow.current_risk
                              )}
                            </div>

                          </td>


                          <td>

                            <span
                              className={`risk-pill ${getRiskClass(
                                cow.risk_category
                              )}`}
                            >

                              {getRiskIcon(
                                cow.risk_category
                              )}

                              {cow.risk_category ||
                                "Unknown"}

                            </span>

                          </td>


                          <td>

                            <span
                              className={`trend-pill ${trend.className}`}
                            >

                              {trend.icon}

                              {trend.label}

                            </span>

                          </td>


                          <td>

                            <div className="forecast-cell">

                              <strong>
                                {formatPercentage(
                                  cow["7d_probability"]
                                )}
                              </strong>

                              {cow["7d_warning"] && (
                                <span className="mini-warning">
                                  Warning
                                </span>
                              )}

                            </div>

                          </td>


                          <td>

                            <div className="forecast-cell">

                              <strong>
                                {formatPercentage(
                                  cow["14d_probability"]
                                )}
                              </strong>

                              {cow["14d_warning"] && (
                                <span className="mini-warning">
                                  Warning
                                </span>
                              )}

                            </div>

                          </td>


                          <td>

                            {cow["7d_warning"] ||
                            cow["14d_warning"] ? (

                              <span className="warning-state">

                                <AlertTriangle size={14} />

                                Elevated

                              </span>

                            ) : (

                              <span className="normal-state">

                                <CheckCircle2 size={14} />

                                No warning

                              </span>

                            )}

                          </td>


                          <td>

                            <button
                              className="action-button"
                              title="Cow profile will be connected next"
                            >
                              Inspect
                              <ChevronRight size={15} />
                            </button>

                          </td>

                        </tr>

                      );

                    })}

                  </tbody>

                </table>

              </div>

            )}

          </div>

        </section>


        {/* =========================
            HIGH RISK COWS
        ========================== */}

        <section className="section">

          <div className="section-heading">

            <div>

              <span className="section-kicker">
                IMMEDIATE ATTENTION
              </span>

              <h2>
                High-Risk Cows
              </h2>

              <p className="section-description">
                Current Model 1 predictions above the backend's
                high-risk threshold.
              </p>

            </div>

          </div>


          <div className="high-risk-grid">

            {herdData?.high_risk_cows?.length > 0 ? (

              herdData.high_risk_cows.map((cow) => (

                <div
                  className="high-risk-card"
                  key={cow.cow_id}
                >

                  <div className="high-risk-card-header">

                    <div className="cow-avatar danger">
                      <HeartPulse size={17} />
                    </div>

                    <div>

                      <strong>
                        {cow.cow_id}
                      </strong>

                      <span>
                        Current Model 1 prediction
                      </span>

                    </div>

                    <AlertTriangle size={19} />

                  </div>


                  <div className="high-risk-value">
                    {formatPercentage(
                      cow.current_risk
                    )}
                  </div>


                  <div className="high-risk-label">
                    Current AI risk
                  </div>


                  <div className="high-risk-forecast-row">

                    <div>

                      <span>
                        7-day
                      </span>

                      <strong>
                        {formatPercentage(
                          cow["7d_probability"]
                        )}
                      </strong>

                    </div>


                    <div>

                      <span>
                        14-day
                      </span>

                      <strong>
                        {formatPercentage(
                          cow["14d_probability"]
                        )}
                      </strong>

                    </div>

                  </div>


                  <button className="inspect-card-button">

                    Prioritize inspection

                    <ChevronRight size={15} />

                  </button>

                </div>

              ))

            ) : (

              <div className="no-high-risk">

                <CheckCircle2 size={24} />

                <div>

                  <strong>
                    No high-risk cows detected
                  </strong>

                  <span>
                    Current Model 1 predictions are below the
                    high-risk threshold.
                  </span>

                </div>

              </div>

            )}

          </div>

        </section>


        {/* =========================
            RECOMMENDATIONS
        ========================== */}

        <section className="section">

          <div className="section-heading">

            <div>

              <span className="section-kicker">
                DECISION SUPPORT
              </span>

              <h2>
                Recommended Actions
              </h2>

              <p className="section-description">
                AI-assisted monitoring recommendations. These are
                decision-support suggestions, not medical diagnoses.
              </p>

            </div>

          </div>


          <div className="recommendation-list">

            {herdData?.recommendations?.length > 0 ? (

              herdData.recommendations.map(
                (recommendation, index) => (

                  <div
                    className="recommendation-card"
                    key={`${recommendation}-${index}`}
                  >

                    <div className="recommendation-icon">
                      {getRecommendationIcon(
                        recommendation
                      )}
                    </div>

                    <div className="recommendation-content">

                      <strong>

                        {index === 0
                          ? "Priority action"
                          : index === 1
                          ? "Monitoring action"
                          : index === 2
                          ? "Forecast review"
                          : "Herd management review"}

                      </strong>

                      <p>
                        {recommendation}
                      </p>

                    </div>

                    <ChevronRight size={18} />

                  </div>

                )
              )

            ) : (

              <div className="recommendation-card">

                <div className="recommendation-icon">
                  <ShieldCheck size={18} />
                </div>

                <div className="recommendation-content">

                  <strong>
                    Routine monitoring
                  </strong>

                  <p>
                    No additional recommendations are currently
                    available from the herd engine.
                  </p>

                </div>

              </div>

            )}

          </div>

        </section>


        {/* =========================
            DATA / SENSOR STATUS
        ========================== */}

        <section className="section">

          <div className="section-heading">

            <div>

              <span className="section-kicker">
                DATA PIPELINE
              </span>

              <h2>
                Monitoring Infrastructure
              </h2>

              <p className="section-description">
                Hardware integration is designed to feed animal and
                environmental observations into the AI pipeline.
              </p>

            </div>

          </div>


          <div className="infrastructure-grid">

            <InfrastructureCard
              icon={<Wifi size={20} />}
              title="ESP32"
              subtitle="IoT controller"
              status="Backend integration ready"
            />

            <InfrastructureCard
              icon={<Database size={20} />}
              title="RFID"
              subtitle="Animal identification"
              status="Integration planned"
            />

            <InfrastructureCard
              icon={<Thermometer size={20} />}
              title="DHT11"
              subtitle="Environmental temperature / humidity"
              status="Integration planned"
            />

            <InfrastructureCard
              icon={<Thermometer size={20} />}
              title="DS18B20"
              subtitle="Temperature sensor"
              status="Integration planned"
            />

            <InfrastructureCard
              icon={<Activity size={20} />}
              title="MPU6050"
              subtitle="Movement / activity signal"
              status="Integration planned"
            />

            <InfrastructureCard
              icon={<Droplets size={20} />}
              title="Milk-quality sensor"
              subtitle="Milk-quality-related signal"
              status="Integration planned"
            />

          </div>


          <div className="sensor-note">

            <Database size={18} />

            <p>
              Sensor values are intentionally not fabricated in this
              prototype dashboard. A sensor will only be shown as
              connected when the backend provides an actual connection
              state or reading.
            </p>

          </div>

        </section>


        {/* =========================
            WORKFLOW
        ========================== */}

        <section className="section">

          <div className="section-heading">

            <div>

              <span className="section-kicker">
                SYSTEM WORKFLOW
              </span>

              <h2>
                From Monitoring to Action
              </h2>

            </div>

          </div>


          <div className="workflow">

            <WorkflowStep
              number="01"
              icon={<Wifi size={19} />}
              title="Monitor"
              description="Collect animal and environmental observations."
            />

            <div className="workflow-line"></div>

            <WorkflowStep
              number="02"
              icon={<Activity size={19} />}
              title="Detect"
              description="Model 1 estimates current individual risk."
            />

            <div className="workflow-line"></div>

            <WorkflowStep
              number="03"
              icon={<TrendingUp size={19} />}
              title="Forecast"
              description="Model 2 estimates future risk probability."
            />

            <div className="workflow-line"></div>

            <WorkflowStep
              number="04"
              icon={<AlertTriangle size={19} />}
              title="Prioritize"
              description="Rank animals requiring closer attention."
            />

            <div className="workflow-line"></div>

            <WorkflowStep
              number="05"
              icon={<ShieldCheck size={19} />}
              title="Act"
              description="Support timely inspection and herd management."
            />

          </div>

        </section>


        {/* =========================
            FUTURE INTEGRATION
        ========================== */}

        <section className="section">

          <div className="future-card">

            <div className="future-icon">
              <Cloud size={23} />
            </div>

            <div className="future-content">

              <span className="section-kicker">
                FUTURE INTEGRATION
              </span>

              <h2>
                Designed to scale beyond the MVP
              </h2>

              <p>
                The architecture can later incorporate additional
                farm records, laboratory results, SCC measurements,
                mobile alerts, multilingual interfaces, cloud
                deployment, GIS-based herd insights and additional
                wearable or IoT signals.
              </p>

            </div>


            <div className="future-tags">

              <span>Planned</span>
              <span>IoT</span>
              <span>Mobile</span>
              <span>Cloud</span>

            </div>

          </div>

        </section>


        {/* =========================
            DISCLAIMER
        ========================== */}

        <footer className="dashboard-footer">

          <div className="footer-brand">

            <HeartPulse size={17} />

            <strong>
              Mastitis AI
            </strong>

          </div>


          <p>
            Prototype decision-support system. AI risk percentages
            are model predictions and should not be interpreted as
            confirmed clinical diagnoses. Veterinary assessment
            remains necessary when clinical signs are present.
          </p>


          <div className="footer-message">

            <Leaf size={15} />

            Early detection • Better monitoring • Smarter dairy management

          </div>

        </footer>

      </main>

    </div>
  );
}


/* ============================================================
   USER ICON
============================================================ */

function UserIcon() {

  return (
    <Users size={16} />
  );
}


/* ============================================================
   RISK DISTRIBUTION
============================================================ */

function RiskDistribution({
  label,
  count,
  total,
  className,
}) {

  const safeCount = Number(count || 0);
  const safeTotal = Number(total || 0);

  const percentage =
    safeTotal > 0
      ? (safeCount / safeTotal) * 100
      : 0;

  return (

    <div className="distribution-item">

      <div className="distribution-item-header">

        <div className="distribution-label">

          <span
            className={`distribution-dot ${className}`}
          ></span>

          {label}

        </div>

        <strong>
          {safeCount}
        </strong>

      </div>


      <div className="distribution-track">

        <div
          className={`distribution-fill ${className}`}
          style={{
            width: `${Math.max(
              percentage,
              safeCount > 0 ? 1 : 0
            )}%`,
          }}
        ></div>

      </div>


      <span className="distribution-percentage">

        {percentage.toFixed(1)}% of herd

      </span>

    </div>
  );
}


/* ============================================================
   FORECAST CARD
============================================================ */

function ForecastCard({
  title,
  value,
  subtitle,
  probability,
}) {

  const numericProbability =
    Number(probability || 0);

  return (

    <div className="forecast-card">

      <div className="forecast-card-header">

        <span>
          {title}
        </span>

        <div className="forecast-card-icon">
          <TrendingUp size={18} />
        </div>

      </div>


      <div className="forecast-value">
        {value}
      </div>


      <div className="forecast-progress">

        <div
          className="forecast-progress-fill"
          style={{
            width: `${Math.min(
              Math.max(
                numericProbability,
                0
              ),
              100
            )}%`,
          }}
        ></div>

      </div>


      <p>
        {subtitle}
      </p>

    </div>
  );
}


/* ============================================================
   INFRASTRUCTURE CARD
============================================================ */

function InfrastructureCard({
  icon,
  title,
  subtitle,
  status,
}) {

  return (

    <div className="infrastructure-card">

      <div className="infrastructure-icon">
        {icon}
      </div>


      <div className="infrastructure-info">

        <strong>
          {title}
        </strong>

        <span>
          {subtitle}
        </span>

      </div>


      <div className="infrastructure-status">

        <span className="planned-dot"></span>

        {status}

      </div>

    </div>
  );
}


/* ============================================================
   WORKFLOW STEP
============================================================ */

function WorkflowStep({
  number,
  icon,
  title,
  description,
}) {

  return (

    <div className="workflow-step">

      <div className="workflow-number">
        {number}
      </div>

      <div className="workflow-icon">
        {icon}
      </div>

      <strong>
        {title}
      </strong>

      <p>
        {description}
      </p>

    </div>
  );
}


export default App;