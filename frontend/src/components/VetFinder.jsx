import { useState } from "react";
import {
    MapPin,
    Phone,
    Navigation,
    ExternalLink,
    Key,
    CheckCircle2,
    Building2
} from "lucide-react";

export default function VetFinder({ t }) {
    const [userLocation, setUserLocation] = useState({
        lat: 10.9613,
        lng: 76.9535,
        address: "Kuniamuthur, Coimbatore, Tamil Nadu 641008"
    });

    const [loadingLoc, setLoadingLoc] = useState(false);
    const [apiKey] = useState(import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "");

    // Localized emergency veterinary directory for Kuniamuthur, Coimbatore
    const coimbatoreVetClinics = [
        {
            id: "v1",
            name: "Government Veterinary Hospital, Kuniamuthur",
            type: "State Government Cattle Hospital",
            distance: "1.2 km away",
            address: "Palakkad Main Road, Kuniamuthur, Coimbatore, Tamil Nadu 641008",
            phone: "+91 422 2252100",
            hours: "Open 24/7 (Emergency Service)",
            rating: 4.8,
            isEmergency: true,
            mapsUrl: "https://www.google.com/maps/search/?api=1&query=Government+Veterinary+Hospital+Kuniamuthur+Coimbatore"
        },
        {
            id: "v2",
            name: "TANUVAS Veterinary University Training & Research Centre",
            type: "University Emergency Veterinary Unit",
            distance: "4.5 km away",
            address: "Saravanampatti / Kalapatti Rd, Coimbatore, Tamil Nadu 641035",
            phone: "+91 422 2669305",
            hours: "8:00 AM - 6:00 PM",
            rating: 4.9,
            isEmergency: true,
            mapsUrl: "https://www.google.com/maps/search/?api=1&query=TANUVAS+Veterinary+University+Coimbatore"
        },
        {
            id: "v3",
            name: "Coimbatore District Co-operative Milk Union (Aavin Vet Services)",
            type: "Dairy Co-operative Mobile Vet Unit",
            distance: "3.8 km away",
            address: "Pachapalayam / Perur Bypass, Coimbatore, Tamil Nadu 641010",
            phone: "+91 422 2341201",
            hours: "24/7 On-Call Mobile Vet Doctor",
            rating: 4.7,
            isEmergency: true,
            mapsUrl: "https://www.google.com/maps/search/?api=1&query=Aavin+Dairy+Coimbatore"
        },
        {
            id: "v4",
            name: "Dr. S. Kumar Cattle Specialty Clinic & Emergency Unit",
            type: "Private Livestock & Bovine Clinic",
            distance: "2.1 km away",
            address: "BK Pudur Road, Kuniamuthur, Coimbatore, Tamil Nadu 641008",
            phone: "+91 98422 11099",
            hours: "7:00 AM - 9:00 PM",
            rating: 4.6,
            isEmergency: false,
            mapsUrl: "https://www.google.com/maps/search/?api=1&query=Veterinary+Doctor+Kuniamuthur+Coimbatore"
        },
        {
            id: "v5",
            name: "Government Veterinary Dispensary, Sundarapuram",
            type: "Government Dairy & Livestock Dispensary",
            distance: "3.0 km away",
            address: "Pollachi Main Rd, Sundarapuram, Coimbatore, Tamil Nadu 641024",
            phone: "+91 422 2672230",
            hours: "8:00 AM - 2:00 PM",
            rating: 4.5,
            isEmergency: false,
            mapsUrl: "https://www.google.com/maps/search/?api=1&query=Veterinary+Dispensary+Sundarapuram+Coimbatore"
        }
    ];

    const handleGeolocate = () => {
        if (!navigator.geolocation) {
            alert("Geolocation is not supported by your browser.");
            return;
        }
        setLoadingLoc(true);
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                setUserLocation({
                    lat: pos.coords.latitude,
                    lng: pos.coords.longitude,
                    address: `GPS: ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)} (Coimbatore Region)`
                });
                setLoadingLoc(false);
            },
            (err) => {
                console.warn("Geolocation error, keeping default Kuniamuthur location:", err);
                setLoadingLoc(false);
            }
        );
    };

    return (
        <div className="vet-finder-page">
            {/* HEADER BAR */}
            <div className="vet-header-bar">
                <div>
                    <h2>{t.vet_finder_title}</h2>
                    <p className="subtitle">{t.vet_finder_subtitle}</p>
                </div>
                <button className="location-btn" onClick={handleGeolocate} disabled={loadingLoc}>
                    <Navigation size={16} className={loadingLoc ? "spin" : ""} />
                    {t.use_my_loc}
                </button>
            </div>

            {/* CURRENT FARM REGION TAG */}
            <div className="current-location-tag glassmorphism">
                <MapPin size={20} className="text-emerald" />
                <div>
                    <span className="loc-label">{t.farm_region}</span>
                    <strong className="loc-address">{userLocation.address}</strong>
                </div>
            </div>


            {/* VET CLINICS DIRECTORY LIST */}
            <div className="vet-list-container">
                {coimbatoreVetClinics.map((clinic) => (
                    <div key={clinic.id} className="vet-card glassmorphism">
                        <div className="vet-card-main">
                            <div className="vet-icon-badge">
                                <Building2 size={24} />
                            </div>

                            <div className="vet-info">
                                <div className="vet-title-row">
                                    <h3>{clinic.name}</h3>
                                    <span className="distance-badge">{clinic.distance}</span>
                                </div>

                                <div className="vet-type">{clinic.type}</div>
                                <div className="vet-address">
                                    <MapPin size={14} /> {clinic.address}
                                </div>
                            </div>
                        </div>

                        <div className="vet-card-actions">
                            <span className="hours-tag">
                                <CheckCircle2 size={14} className="text-emerald" /> {clinic.hours}
                            </span>

                            <div className="action-buttons-row">
                                <a href={`tel:${clinic.phone}`} className="phone-btn">
                                    <Phone size={14} /> {t.call_doctor} ({clinic.phone})
                                </a>
                                <a href={clinic.mapsUrl} target="_blank" rel="noopener noreferrer" className="maps-btn">
                                    <ExternalLink size={14} /> {t.open_maps}
                                </a>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
