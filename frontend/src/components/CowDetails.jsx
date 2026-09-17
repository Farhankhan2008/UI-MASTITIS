import { useMemo } from "react";
import {
    ArrowLeft,
    Activity,
    Droplets,
    Thermometer,
    Wifi,
    AlertTriangle,
    CheckCircle2,
    HeartPulse,
    IndianRupee,
    ShieldAlert,
    Stethoscope
} from "lucide-react";

export default function CowDetails({ cow, onBack, onNavigateToVets, t }) {
    if (!cow) return null;

    const historyData = useMemo(() => {
        let parsed = [];
        if (cow.history_json) {
            try {
                parsed = typeof cow.history_json === "string" ? JSON.parse(cow.history_json) : cow.history_json;
            } catch (e) {
                console.error("Error parsing history_json:", e);
            }
        }

        if (!parsed || parsed.length === 0) {
            const baseRisk = Math.max(5, cow.current_risk - 25);
            for (let d = -14; d <= 0; d++) {
                const val = baseRisk + ((cow.current_risk - baseRisk) / 14) * (d + 14);
                parsed.push({ day: `Day ${d}`, day_num: d, risk: Number(val.toFixed(1)) });
            }
        }
        return parsed;
    }, [cow]);

    const forecastPoints = useMemo(() => {
        const current = Number(cow.current_risk || 0);
        const r24h = Number(cow.risk_24h !== undefined ? cow.risk_24h : current);
        const r48h = Number(cow.risk_48h !== undefined ? cow.risk_48h : r24h);
        const r3d = Number(cow.risk_3d !== undefined ? cow.risk_3d : r48h);
        const r7d = Number(cow["7d_probability"] !== undefined ? cow["7d_probability"] : r3d);
        const r14d = Number(cow["14d_probability"] !== undefined ? cow["14d_probability"] : r7d);

        return [
            { label: "Day 0 (Now)", day_num: 0, risk: current },
            { label: "+24 Hours", day_num: 1, risk: r24h },
            { label: "+48 Hours", day_num: 2, risk: r48h },
            { label: "+3 Days", day_num: 3, risk: r3d },
            { label: "+7 Days", day_num: 7, risk: r7d },
            { label: "+14 Days", day_num: 14, risk: r14d },
        ];
    }, [cow]);

    // Financial Loss Calculation in ₹ INR
    const financialAnalysis = useMemo(() => {
        const milkPricePerLiter = 42; // ₹42/Liter average rate
        const currentYield = Number(cow.milk_yield_liters || 15);
        const risk = Number(cow.current_risk || 0);

        let yieldDropLitersPerDay = 0;
        let treatmentCost = 0;
        let milkRejectionCost = 0;

        if (risk >= 70) {
            yieldDropLitersPerDay = Math.round(currentYield * 0.45);
            milkRejectionCost = currentYield * milkPricePerLiter;
            treatmentCost = 1200;
        } else if (risk >= 40) {
            yieldDropLitersPerDay = Math.round(currentYield * 0.25);
            milkRejectionCost = (currentYield * 0.5) * milkPricePerLiter;
            treatmentCost = 600;
        } else if (risk >= 20) {
            yieldDropLitersPerDay = Math.round(currentYield * 0.10);
            milkRejectionCost = 0;
            treatmentCost = 250;
        } else {
            yieldDropLitersPerDay = 0;
            milkRejectionCost = 0;
            treatmentCost = 0;
        }

        const dailyMilkLossRupees = Math.round(yieldDropLitersPerDay * milkPricePerLiter + milkRejectionCost);
        const totalWeeklyLossRupees = (dailyMilkLossRupees * 7) + treatmentCost;

        return {
            milkPricePerLiter,
            yieldDropLitersPerDay,
            dailyMilkLossRupees,
            treatmentCost,
            totalWeeklyLossRupees
        };
    }, [cow]);

    // Combined chart calculation for SVG line rendering
    const chartWidth = 720;
    const chartHeight = 260;
    const paddingLeft = 50;
    const paddingRight = 30;
    const paddingTop = 20;
    const paddingBottom = 40;

    const usableWidth = chartWidth - paddingLeft - paddingRight;
    const usableHeight = chartHeight - paddingTop - paddingBottom;

    const pastCoords = historyData.map((pt) => {
        const day = pt.day_num !== undefined ? pt.day_num : -14;
        const x = paddingLeft + ((day + 14) / 28) * usableWidth;
        const y = paddingTop + (1 - pt.risk / 100) * usableHeight;
        return { x, y, day: pt.day, risk: pt.risk };
    });

    const futureCoords = forecastPoints.map((pt) => {
        const x = paddingLeft + ((pt.day_num + 14) / 28) * usableWidth;
        const y = paddingTop + (1 - pt.risk / 100) * usableHeight;
        return { x, y, label: pt.label, risk: pt.risk };
    });

    const pastSvgPoints = pastCoords.map((c) => `${c.x},${c.y}`).join(" ");
    const futureSvgPoints = futureCoords.map((c) => `${c.x},${c.y}`).join(" ");

    const isHigh = cow.risk_category?.toLowerCase().includes("high") || cow.current_risk >= 70;
    const isMod = cow.risk_category?.toLowerCase().includes("moderate") || (cow.current_risk >= 40 && cow.current_risk < 70);
    const isLow = cow.risk_category?.toLowerCase().includes("low") || (cow.current_risk >= 20 && cow.current_risk < 40);

    return (
        <div className="cow-details-page">
            {/* HEADER BAR */}
            <div className="details-header">
                <button className="back-button" onClick={onBack}>
                    <ArrowLeft size={18} /> {t.back_overview}
                </button>
                <div className="cow-title-block">
                    <h2>{cow.cow_id} {t.analytics_title}</h2>
                    <span className={`risk-badge ${isHigh ? "badge-high" : isMod ? "badge-mod" : isLow ? "badge-low" : "badge-healthy"}`}>
                        {cow.risk_category}
                    </span>
                </div>
            </div>

            {/* RECOMMENDED STEPS TO TAKE BASED ON RISK */}
            <div className={`action-recommendation-card glassmorphism ${isHigh ? "rec-high" : isMod ? "rec-mod" : isLow ? "rec-low" : "rec-healthy"}`}>
                <div className="rec-header">
                    {isHigh ? (
                        <AlertTriangle size={24} className="text-red" />
                    ) : isMod ? (
                        <ShieldAlert size={24} className="text-amber" />
                    ) : (
                        <CheckCircle2 size={24} className="text-emerald" />
                    )}
                    <div>
                        <h3>{t.rec_steps_title} ({cow.cow_id})</h3>
                        <span className="rec-subtitle">{t.custom_guidance} ({cow.current_risk}%)</span>
                    </div>
                </div>

                <div className="rec-content">
                    {isHigh ? (
                        <div>
                            <p>
                                <strong>CRITICAL VETERINARY ACTION REQUIRED:</strong> Isolate {cow.cow_id} immediately from the main milking pipeline to avoid cross-contamination.
                                Milk electrical conductivity ({cow.milk_conductivity_ms_cm} mS/cm) indicates active udder inflammation.
                            </p>
                            <button className="goto-vet-btn" onClick={onNavigateToVets}>
                                <Stethoscope size={16} /> {t.find_vet_btn}
                            </button>
                        </div>
                    ) : isMod ? (
                        <div>
                            <p>
                                <strong>INCREASED MONITORING &amp; MILK SEGREGATION:</strong> Separate milk yield monitoring for the next 48 hours.
                                Palpate udder quarters for swelling or localized heat. Schedule a routine veterinary checkup if risk increases beyond 50%.
                            </p>
                            <button className="goto-vet-btn secondary" onClick={onNavigateToVets}>
                                <Stethoscope size={16} /> {t.view_vet_btn}
                            </button>
                        </div>
                    ) : isLow ? (
                        <p>
                            <strong>SANITATION &amp; EQUIPMENT REVIEW:</strong> Inspect milking machine cluster vacuum pressure and teat liner condition.
                            Ensure post-milking teat dipping is conducted thoroughly with 0.5% iodine solution.
                        </p>
                    ) : (
                        <p>
                            <strong>PREVENTION &amp; BEST PRACTICES:</strong> Cow is healthy with optimal milk parameters ({cow.milk_yield_liters} L/day).
                            Maintain dry, clean bedding in housing stalls and continue routine daily AI telemetry tracking.
                        </p>
                    )}
                </div>
            </div>

            {/* FINANCIAL IMPACT & DELAYED TREATMENT AFTER-EFFECTS (IN RUPEES) */}
            <div className="financial-card glassmorphism">
                <div className="financial-header">
                    <div className="fin-icon-badge">
                        <IndianRupee size={22} />
                    </div>
                    <div>
                        <h3>{t.fin_title}</h3>
                        <p>{t.fin_desc}</p>
                    </div>
                </div>

                <div className="financial-body-grid">
                    {/* COST METRICS */}
                    <div className="cost-breakdown-box">
                        <div className="cost-item">
                            <span>{t.milk_price}</span>
                            <strong>₹{financialAnalysis.milkPricePerLiter} / Liter</strong>
                        </div>

                        <div className="cost-item">
                            <span>{t.daily_yield_drop}</span>
                            <strong className={isHigh ? "text-red" : "text-emerald"}>-{financialAnalysis.yieldDropLitersPerDay} Liters / day</strong>
                        </div>

                        <div className="cost-item">
                            <span>{t.daily_loss_rejection}</span>
                            <strong className="text-red">₹{financialAnalysis.dailyMilkLossRupees} / day</strong>
                        </div>

                        <div className="cost-item highlight-cost">
                            <span>{t.potential_weekly_loss}</span>
                            <strong className="total-rupees">₹{financialAnalysis.totalWeeklyLossRupees.toLocaleString("en-IN")}</strong>
                        </div>
                    </div>

                    {/* AFTER-EFFECTS OF DELAYED TREATMENT */}
                    <div className="effects-box">
                        <h4>{t.clinical_consequences}</h4>
                        <ul>
                            <li>{t.effect_1}</li>
                            <li>{t.effect_2}</li>
                            <li>{t.effect_3}</li>
                            <li>{t.effect_4}</li>
                        </ul>
                    </div>
                </div>
            </div>

            {/* 4 PREDICTION HORIZONS CARDS */}
            <div className="forecast-4-grid">
                <div className="forecast-card glassmorphism">
                    <div className="forecast-period">24 HOURS FORECAST</div>
                    <div className={`forecast-risk ${cow.risk_24h > 50 ? "text-red" : "text-emerald"}`}>
                        {Number(cow.risk_24h || cow.current_risk).toFixed(1)}%
                    </div>
                    <span className="forecast-sub">Model 2 Instant Neural Risk</span>
                </div>

                <div className="forecast-card glassmorphism">
                    <div className="forecast-period">48 HOURS FORECAST</div>
                    <div className={`forecast-risk ${cow.risk_48h > 50 ? "text-red" : "text-emerald"}`}>
                        {Number(cow.risk_48h || cow.current_risk).toFixed(1)}%
                    </div>
                    <span className="forecast-sub">Short-term Udder Health</span>
                </div>

                <div className="forecast-card glassmorphism">
                    <div className="forecast-period">3 DAYS FORECAST</div>
                    <div className={`forecast-risk ${cow.risk_3d > 50 ? "text-red" : "text-emerald"}`}>
                        {Number(cow.risk_3d || cow.current_risk).toFixed(1)}%
                    </div>
                    <span className="forecast-sub">Medium Trend Horizon</span>
                </div>

                <div className="forecast-card glassmorphism">
                    <div className="forecast-period">7 TO 14 DAYS FORECAST</div>
                    <div className={`forecast-risk ${cow["14d_probability"] > 50 ? "text-red" : "text-emerald"}`}>
                        {Number(cow["14d_probability"] || cow["7d_probability"]).toFixed(1)}%
                    </div>
                    <span className="forecast-sub">7d: {Number(cow["7d_probability"]).toFixed(1)}% | 14d: {Number(cow["14d_probability"]).toFixed(1)}%</span>
                </div>
            </div>

            {/* CONTINUOUS vs DOTTED RISK TREND GRAPH */}
            <div className="chart-card glassmorphism">
                <div className="chart-header">
                    <div>
                        <h3>{t.risk_trend_proj}</h3>
                        <p>{t.past_future_desc}</p>
                    </div>
                    <div className="chart-legend">
                        <div className="legend-item">
                            <span className="legend-line solid"></span>
                            <span>{t.past_trend}</span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-line dotted"></span>
                            <span>{t.next_forecast}</span>
                        </div>
                    </div>
                </div>

                {/* SVG COMBINED LINE GRAPH */}
                <div className="svg-container">
                    <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="risk-chart-svg">
                        {[0, 25, 50, 75, 100].map((val) => {
                            const y = paddingTop + (1 - val / 100) * usableHeight;
                            return (
                                <g key={val}>
                                    <line x1={paddingLeft} y1={y} x2={chartWidth - paddingRight} y2={y} stroke="#e2e8f0" strokeDasharray="3 3" />
                                    <text x={paddingLeft - 10} y={y + 4} textAnchor="end" fontSize="10" fill="#94a3b8">
                                        {val}%
                                    </text>
                                </g>
                            );
                        })}

                        {(() => {
                            const x0 = paddingLeft + (14 / 28) * usableWidth;
                            return (
                                <g>
                                    <line x1={x0} y1={paddingTop} x2={x0} y2={chartHeight - paddingBottom} stroke="#047857" strokeWidth="2" strokeDasharray="4 4" />
                                    <text x={x0} y={chartHeight - paddingBottom + 18} textAnchor="middle" fontSize="11" fontWeight="700" fill="#047857">
                                        TODAY
                                    </text>
                                </g>
                            );
                        })()}

                        <polyline points={pastSvgPoints} fill="none" stroke="#047857" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />

                        {pastCoords.map((pt, idx) => (
                            <circle key={idx} cx={pt.x} cy={pt.y} r="4" fill="#047857" stroke="#ffffff" strokeWidth="2">
                                <title>{`${pt.day}: ${pt.risk}%`}</title>
                            </circle>
                        ))}

                        <polyline points={futureSvgPoints} fill="none" stroke="#dc2626" strokeWidth="3.5" strokeDasharray="6 6" strokeLinecap="round" strokeLinejoin="round" />

                        {futureCoords.map((pt, idx) => (
                            <circle key={idx} cx={pt.x} cy={pt.y} r="4" fill="#dc2626" stroke="#ffffff" strokeWidth="2">
                                <title>{`${pt.label}: ${pt.risk}%`}</title>
                            </circle>
                        ))}
                    </svg>
                </div>
            </div>

            {/* 6 SENSOR TELEMETRY VITALS GRID */}
            <div className="vitals-grid">
                <div className="vital-card glassmorphism">
                    <Droplets size={20} className="vital-icon text-emerald" />
                    <div>
                        <span className="vital-label">Milk Yield</span>
                        <strong className="vital-val">{cow.milk_yield_liters} L/day</strong>
                    </div>
                </div>

                <div className="vital-card glassmorphism">
                    <Activity size={20} className="vital-icon text-amber" />
                    <div>
                        <span className="vital-label">Milk Conductivity</span>
                        <strong className="vital-val">{cow.milk_conductivity_ms_cm} mS/cm</strong>
                    </div>
                </div>

                <div className="vital-card glassmorphism">
                    <HeartPulse size={20} className="vital-icon text-blue" />
                    <div>
                        <span className="vital-label">Cow Activity Index</span>
                        <strong className="vital-val">{cow.cow_activity} steps/hr</strong>
                    </div>
                </div>

                <div className="vital-card glassmorphism">
                    <Thermometer size={20} className="vital-icon text-red" />
                    <div>
                        <span className="vital-label">Milk Temperature</span>
                        <strong className="vital-val">{cow.milk_temperature_c} °C</strong>
                    </div>
                </div>

                <div className="vital-card glassmorphism">
                    <Thermometer size={20} className="vital-icon text-slate" />
                    <div>
                        <span className="vital-label">Barn Temperature</span>
                        <strong className="vital-val">{cow.environment_temperature_c} °C</strong>
                    </div>
                </div>

                <div className="vital-card glassmorphism">
                    <Wifi size={20} className="vital-icon text-emerald" />
                    <div>
                        <span className="vital-label">Humidity</span>
                        <strong className="vital-val">{cow.humidity_percent} %</strong>
                    </div>
                </div>
            </div>
        </div>
    );
}
