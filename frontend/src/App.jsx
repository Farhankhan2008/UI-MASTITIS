import { useState, useEffect, useMemo } from "react";
import Login from "./components/Login";
import CowDetails from "./components/CowDetails";
import VetFinder from "./components/VetFinder";
import { translations } from "./i18n/translations";
import { getHerdRisk } from "./services/api";
import {
  ShieldAlert,
  Activity,
  Search,
  RefreshCw,
  LogOut,
  LayoutDashboard,
  Stethoscope,
  Settings,
  AlertTriangle,
  AlertCircle,
  Eye,
  CheckCircle2,
  Globe,
  X
} from "lucide-react";
import "./App.css";

export default function App() {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem("mastitis_user");
    return saved ? JSON.parse(saved) : null;
  });

  const [herdData, setHerdData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCowId, setSelectedCowId] = useState(null);
  const [activeView, setActiveView] = useState("herd"); // 'herd' | 'cow_details' | 'vet_finder'
  const [activeSection, setActiveSection] = useState(null); // 'sec-priority-high' | 'sec-mod-attention' | 'sec-under-monitor' | 'sec-healthy-cows'
  const [inspectedCows, setInspectedCows] = useState(new Set());

  // Multilingual state (default: English)
  const [lang, setLang] = useState(() => localStorage.getItem("mastitis_lang") || "en");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const t = translations[lang] || translations.en;

  const handleCategoryClick = (sectionId) => {
    setActiveView("herd");
    setSelectedCowId(null);
    setActiveSection(sectionId);
    setTimeout(() => {
      const el = document.getElementById(sectionId);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }, 50);
  };

  const handleSetLang = (newLang) => {
    setLang(newLang);
    localStorage.setItem("mastitis_lang", newLang);
  };

  const fetchHerd = async () => {
    setLoading(true);
    try {
      const data = await getHerdRisk();
      if (data && data.priority_cows) {
        setHerdData(data.priority_cows);
      }
    } catch (e) {
      console.error("Failed to connect to backend engine:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (user) {
      fetchHerd();
    }
  }, [user]);

  // Scroll-Spy Effect: Automatically update sidebar active item as user scrolls
  useEffect(() => {
    if (activeView !== "herd" || selectedCowId) return;

    const sectionIds = [
      "sec-priority-high",
      "sec-mod-attention",
      "sec-under-monitor",
      "sec-healthy-cows"
    ];

    const handleScroll = () => {
      const scrollPos = window.scrollY;

      // Top of page highlights Herd Overview
      if (scrollPos < 260) {
        setActiveSection(null);
        return;
      }

      let currentSection = null;
      for (let i = sectionIds.length - 1; i >= 0; i--) {
        const el = document.getElementById(sectionIds[i]);
        if (el) {
          const top = el.offsetTop - 160;
          if (scrollPos >= top) {
            currentSection = sectionIds[i];
            break;
          }
        }
      }

      if (currentSection) {
        setActiveSection(currentSection);
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, [activeView, selectedCowId]);

  const handleLogin = (userData) => {
    setUser(userData);
    localStorage.setItem("mastitis_user", JSON.stringify(userData));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem("mastitis_user");
  };

  // Restructured cow groups by risk tier
  const priorityHighCows = useMemo(() => {
    return herdData.filter((c) => c.risk_category === "High Risk" || c.current_risk >= 70);
  }, [herdData]);

  const modDiseaseCows = useMemo(() => {
    return herdData.filter((c) => c.risk_category === "Moderate Risk" || (c.current_risk >= 40 && c.current_risk < 70));
  }, [herdData]);

  const underMonitorCows = useMemo(() => {
    return herdData.filter((c) => c.risk_category === "Low Risk" || (c.current_risk >= 20 && c.current_risk < 40));
  }, [herdData]);

  const healthyCows = useMemo(() => {
    return herdData.filter((c) => c.risk_category === "Healthy" || c.current_risk < 20);
  }, [herdData]);

  // Filter search handling
  const isSearchActive = searchTerm.trim().length > 0;
  const searchedCow = useMemo(() => {
    if (!isSearchActive) return null;
    const clean = searchTerm.trim().toUpperCase();
    return herdData.find((c) => c.cow_id.toUpperCase() === clean);
  }, [searchTerm, herdData, isSearchActive]);

  const isCowNotFound = isSearchActive && !searchedCow;

  const selectedCow = useMemo(() => {
    if (!selectedCowId) return null;
    return herdData.find((c) => c.cow_id === selectedCowId);
  }, [selectedCowId, herdData]);

  const toggleInspection = (id, e) => {
    e.stopPropagation();
    setInspectedCows((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  const renderCowCard = (cow) => {
    const isHigh = cow.risk_category === "High Risk" || cow.current_risk >= 70;
    const isMod = cow.risk_category === "Moderate Risk" || (cow.current_risk >= 40 && cow.current_risk < 70);
    const isLow = cow.risk_category === "Low Risk" || (cow.current_risk >= 20 && cow.current_risk < 40);

    return (
      <div
        key={cow.cow_id}
        className={`cow-card glassmorphism ${isHigh ? "card-high" : isMod ? "card-mod" : isLow ? "card-low" : "card-healthy"}`}
        onClick={() => {
          setSelectedCowId(cow.cow_id);
          setActiveView("cow_details");
        }}
      >
        <div className="cow-card-header">
          <div className="cow-identity">
            <div className="cow-avatar">
              <Activity size={18} />
            </div>
            <div>
              <h4>{cow.cow_id}</h4>
              <span className="cow-breed">{cow.breed}</span>
            </div>
          </div>
          <span className={`risk-badge ${isHigh ? "badge-high" : isMod ? "badge-mod" : isLow ? "badge-low" : "badge-healthy"}`}>
            {cow.risk_category}
          </span>
        </div>

        <div className="risk-meter-container">
          <div className="risk-meter-header">
            <span>Risk Score</span>
            <strong>{cow.current_risk}%</strong>
          </div>
          <div className="risk-track">
            <div
              className={`risk-fill ${isHigh ? "fill-high" : isMod ? "fill-mod" : isLow ? "fill-low" : "fill-healthy"}`}
              style={{ width: `${Math.min(100, cow.current_risk)}%` }}
            ></div>
          </div>
        </div>

        {/* 4 PREDICTION HORIZONS */}
        <div className="prediction-grid-4">
          <div className="p-box">
            <span className="p-label">24h</span>
            <span className="p-val">{cow.risk_24h || cow.current_risk}%</span>
          </div>
          <div className="p-box">
            <span className="p-label">48h</span>
            <span className="p-val">{cow.risk_48h || cow.risk_24h}%</span>
          </div>
          <div className="p-box">
            <span className="p-label">3d</span>
            <span className="p-val">{cow.risk_3d || cow.risk_48h}%</span>
          </div>
          <div className="p-box">
            <span className="p-label">14d</span>
            <span className="p-val">{cow["14d_probability"] || cow["7d_probability"]}%</span>
          </div>
        </div>

        <div className="cow-card-footer">
          <button
            className={`inspection-btn ${inspectedCows.has(cow.cow_id) ? "done" : ""}`}
            onClick={(e) => toggleInspection(cow.cow_id, e)}
          >
            {inspectedCows.has(cow.cow_id) ? "Inspected" : "Mark Inspected"}
          </button>
          <span className="details-link">{t.graph_details}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="app">
      {/* SIDEBAR NAVIGATION - ONLY 2 MAIN SUBPAGES + SETTINGS */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <ShieldAlert size={26} />
          </div>
          <div>
            <div className="brand-name">{t.brand_name}</div>
            <div className="brand-subtitle">{t.brand_subtitle}</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-title">{t.herd_monitoring}</div>
          <button
            className={`nav-item ${activeView === "herd" && !activeSection ? "active" : ""}`}
            onClick={() => {
              setActiveView("herd");
              setSelectedCowId(null);
              setActiveSection(null);
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          >
            <LayoutDashboard size={18} />
            <span>{t.herd_overview}</span>
          </button>

          {/* 1. HIGH RISK / PRIORITY HIGH */}
          <button
            className={`nav-item ${activeSection === "sec-priority-high" ? "active" : ""}`}
            onClick={() => handleCategoryClick("sec-priority-high")}
          >
            <AlertTriangle size={18} className="text-red" />
            <span>{t.priority_cows_title}</span>
            <span className="sidebar-badge badge-high">{priorityHighCows.length}</span>
          </button>

          {/* 2. MODERATE RISK / NEED ATTENTION */}
          <button
            className={`nav-item ${activeSection === "sec-mod-attention" ? "active" : ""}`}
            onClick={() => handleCategoryClick("sec-mod-attention")}
          >
            <AlertCircle size={18} className="text-amber" />
            <span>{t.mod_disease_title}</span>
            <span className="sidebar-badge badge-mod">{modDiseaseCows.length}</span>
          </button>

          {/* 3. LOW RISK / UNDER MONITOR */}
          <button
            className={`nav-item ${activeSection === "sec-under-monitor" ? "active" : ""}`}
            onClick={() => handleCategoryClick("sec-under-monitor")}
          >
            <Eye size={18} className="text-blue" />
            <span>{t.under_monitor_title}</span>
            <span className="sidebar-badge badge-low">{underMonitorCows.length}</span>
          </button>

          {/* 4. HEALTHY COWS */}
          <button
            className={`nav-item ${activeSection === "sec-healthy-cows" ? "active" : ""}`}
            onClick={() => handleCategoryClick("sec-healthy-cows")}
          >
            <CheckCircle2 size={18} className="text-emerald" />
            <span>{t.healthy_cows_title}</span>
            <span className="sidebar-badge badge-healthy">{healthyCows.length}</span>
          </button>

          {/* NEARBY VET DOCTORS */}
          <button
            className={`nav-item ${activeView === "vet_finder" ? "active" : ""}`}
            onClick={() => {
              setActiveView("vet_finder");
              setSelectedCowId(null);
              setActiveSection(null);
            }}
          >
            <Stethoscope size={18} />
            <span>{t.nearby_vets}</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <button className="nav-item settings-bottom-btn" onClick={() => setIsSettingsOpen(true)}>
            <Settings size={18} />
            <span>{t.settings}</span>
          </button>

          <div className="user-profile-badge">
            <div className="user-avatar">{(user.name || user.username || "F").charAt(0).toUpperCase()}</div>
            <div className="user-info">
              <span className="user-name">{user.name || user.username || "Farmer"}</span>
              <span className="user-role">{user.role || "farmer"}</span>
            </div>
            <button className="logout-btn" onClick={handleLogout} title="Logout">
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="main-content">
        {/* TOPBAR */}
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              {t.brand_name.toUpperCase()} /{" "}
              {activeView === "cow_details" ? "COW ANALYTICS" : activeView === "vet_finder" ? "VET DIRECTORY" : "HERD OVERVIEW"}
            </div>
            <h1>
              {activeView === "cow_details"
                ? `${selectedCowId} Analytics`
                : activeView === "vet_finder"
                  ? t.vet_finder_title
                  : t.herd_overview}
            </h1>
          </div>

          <div className="topbar-actions">
            {/* LANGUAGE INDICATOR */}
            <button className="lang-indicator-btn" onClick={() => setIsSettingsOpen(true)}>
              <Globe size={16} /> {lang.toUpperCase()}
            </button>

            <div className="system-status">
              <span className="online-dot"></span>
              {t.cows_online}
            </div>

            <button className="refresh-button" onClick={fetchHerd} disabled={loading}>
              <RefreshCw size={14} className={loading ? "spin" : ""} />
              {t.refresh_data}
            </button>
          </div>
        </header>

        {/* VIEW ROUTING */}
        {activeView === "cow_details" && selectedCow ? (
          <CowDetails
            cow={selectedCow}
            lang={lang}
            t={t}
            onBack={() => {
              setActiveView("herd");
              setSelectedCowId(null);
            }}
            onNavigateToVets={() => {
              setActiveView("vet_finder");
              setSelectedCowId(null);
            }}
          />
        ) : activeView === "vet_finder" ? (
          <VetFinder lang={lang} t={t} />
        ) : (
          <div>
            {/* SUMMARY METRICS HEADER - SHOWS ONLY HERD RISK %, TOTAL COWS, & BREAKDOWN */}
            <div className="section">
              <div className="hero-status-grid">
                <div className="herd-status-card glassmorphism status-high featured-risk-card">
                  <div className="status-card-top">
                    <span className="featured-badge"><AlertTriangle size={14} /> {t.herd_risk_level}</span>
                    <span className="risk-tag-high">CRITICAL ALERT</span>
                  </div>
                  <div className="status-value-row">
                    <div className="status-value">HIGH (78%)</div>
                    <div className="status-sub-metric">{priorityHighCows.length} Cows At Risk</div>
                  </div>
                  <div className="status-progress-track">
                    <div className="status-progress-fill" style={{ width: '78%' }}></div>
                  </div>
                  <div className="status-caption">
                    {priorityHighCows.length} {t.herd_risk_desc}
                  </div>
                </div>

                <div className="metric-card glassmorphism card-shadow-grey">
                  <span className="metric-label">{t.total_herd}</span>
                  <div className="metric-value">100</div>
                  <span className="metric-caption">{t.monitored_cows}</span>
                </div>

                <div className="metric-card glassmorphism card-shadow-red">
                  <span className="metric-label">{t.high_risk}</span>
                  <div className="metric-value text-red">{priorityHighCows.length}</div>
                  <span className="metric-caption">Risk ≥ 70%</span>
                </div>

                <div className="metric-card glassmorphism card-shadow-yellow">
                  <span className="metric-label">{t.mod_risk}</span>
                  <div className="metric-value text-amber">{modDiseaseCows.length}</div>
                  <span className="metric-caption">40% - 69% Risk</span>
                </div>

                <div className="metric-card glassmorphism card-shadow-blue">
                  <span className="metric-label">{t.low_risk}</span>
                  <div className="metric-value text-blue">{underMonitorCows.length}</div>
                  <span className="metric-caption">20% - 39% Risk</span>
                </div>

                <div className="metric-card glassmorphism card-shadow-green">
                  <span className="metric-label">{t.healthy_label}</span>
                  <div className="metric-value text-emerald">{healthyCows.length}</div>
                  <span className="metric-caption">Risk &lt; 20%</span>
                </div>
              </div>
            </div>

            {/* SEARCH & FILTER BAR FOR ALL 100 COWS */}
            <div className="section">
              <div className="filter-bar glassmorphism">
                <div className="search-box">
                  <Search size={18} className="text-muted" />
                  <input
                    type="text"
                    placeholder={t.search_placeholder}
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                </div>
              </div>

              {/* NON-EXISTENT COW ERROR BANNER */}
              {isCowNotFound && (
                <div className="not-found-banner glassmorphism">
                  <AlertTriangle size={24} className="text-red" />
                  <div>
                    <h3>Cow &quot;{searchTerm.trim()}&quot; {t.not_exist_title}</h3>
                    <p>{t.not_exist_desc}</p>
                  </div>
                </div>
              )}
            </div>

            {/* IF SEARCHING SPECIFIC COW */}
            {isSearchActive && searchedCow ? (
              <div className="section">
                <div className="section-heading">
                  <h2>Search Result for {searchedCow.cow_id}</h2>
                </div>
                <div className="cow-grid">{renderCowCard(searchedCow)}</div>
              </div>
            ) : !isSearchActive ? (
              /* RESTRUCTURED 4 CATEGORIZED SECTIONS */
              <div className="section">
                {/* 1. PRIORITY COWS SECTION (HIGH RISK) */}
                <div id="sec-priority-high" className="herd-sub-section">
                  <div className="section-heading">
                    <span className="section-kicker text-red">🔴 CATEGORY 1</span>
                    <h2>{t.priority_cows_title} ({priorityHighCows.length})</h2>
                    <p className="page-description">{t.priority_cows_desc}</p>
                  </div>
                  <div className="cow-grid">
                    {priorityHighCows.map((cow) => renderCowCard(cow))}
                  </div>
                </div>

                {/* 2. HIGH POTENTIAL FOR DISEASE SECTION (MODERATE RISK) */}
                <div id="sec-mod-attention" className="herd-sub-section margin-top-lg">
                  <div className="section-heading">
                    <span className="section-kicker text-amber">🟠 CATEGORY 2</span>
                    <h2>{t.mod_disease_title} ({modDiseaseCows.length})</h2>
                    <p className="page-description">{t.mod_disease_desc}</p>
                  </div>
                  <div className="cow-grid">
                    {modDiseaseCows.map((cow) => renderCowCard(cow))}
                  </div>
                </div>

                {/* 3. UNDER MONITOR SECTION (LOW RISK) */}
                <div id="sec-under-monitor" className="herd-sub-section margin-top-lg">
                  <div className="section-heading">
                    <span className="section-kicker text-blue">🟡 CATEGORY 3</span>
                    <h2>{t.under_monitor_title} ({underMonitorCows.length})</h2>
                    <p className="page-description">{t.under_monitor_desc}</p>
                  </div>
                  <div className="cow-grid">
                    {underMonitorCows.map((cow) => renderCowCard(cow))}
                  </div>
                </div>

                {/* 4. HEALTHY COWS SECTION */}
                <div id="sec-healthy-cows" className="herd-sub-section margin-top-lg">
                  <div className="section-heading">
                    <span className="section-kicker text-emerald">🟢 CATEGORY 4</span>
                    <h2>{t.healthy_cows_title} ({healthyCows.length})</h2>
                    <p className="page-description">{t.healthy_cows_desc}</p>
                  </div>
                  <div className="cow-grid">
                    {healthyCows.map((cow) => renderCowCard(cow))}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        )}

        {/* SETTINGS MODAL FOR 7 INDIAN LANGUAGES */}
        {isSettingsOpen && (
          <div className="modal-overlay" onClick={() => setIsSettingsOpen(false)}>
            <div className="modal-card glassmorphism" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>{t.settings_title}</h3>
                <button className="close-btn" onClick={() => setIsSettingsOpen(false)}>
                  <X size={20} />
                </button>
              </div>

              <div className="modal-body">
                <label className="setting-label">{t.select_language}</label>
                <div className="lang-grid">
                  {[
                    { code: "en", label: "English" },
                    { code: "ta", label: "தமிழ் (Tamil)" },
                    { code: "hi", label: "हिन्दी (Hindi)" },
                    { code: "te", label: "తెలుగు (Telugu)" },
                    { code: "ml", label: "മലയാളം (Malayalam)" },
                    { code: "kn", label: "ಕನ್ನಡ (Kannada)" },
                    { code: "mr", label: "मराठी (Marathi)" },
                  ].map((item) => (
                    <button
                      key={item.code}
                      className={`lang-option-btn ${lang === item.code ? "active" : ""}`}
                      onClick={() => handleSetLang(item.code)}
                    >
                      <Globe size={16} />
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="modal-footer">
                <button className="save-btn" onClick={() => setIsSettingsOpen(false)}>
                  {t.save_close}
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}