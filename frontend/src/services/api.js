const API_BASE_URL = "http://127.0.0.1:8000";


async function request(endpoint, options = {}) {
  const token = localStorage.getItem("token");

  const headers = {
    ...(options.headers || {})
  };

  // Add JWT token for protected API requests
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  // Add JSON content type when sending JSON
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    {
      ...options,
      headers
    }
  );

  let data;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {

    // Token expired or invalid
    if (response.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("username");
      localStorage.removeItem("role");
    }

    throw new Error(
      data?.detail || `API request failed: ${response.status}`
    );
  }

  return data;
}


// ============================================================
// LOGIN
// ============================================================

export async function loginUser(username, password) {

  const formData = new URLSearchParams();

  formData.append("username", username);
  formData.append("password", password);

  const response = await fetch(
    `${API_BASE_URL}/login`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: formData
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data?.detail || `Login failed: ${response.status}`
    );
  }

  return data;
}


// ============================================================
// REGISTER
// ============================================================

export async function registerUser(
  username,
  password,
  role = "farmer"
) {

  return request(
    "/register",
    {
      method: "POST",
      body: JSON.stringify({
        username,
        password,
        role
      })
    }
  );
}


// ============================================================
// HEALTH
// ============================================================

export async function getHealth() {
  return request("/health");
}


// ============================================================
// COWS
// ============================================================

export async function getCows() {
  return request("/cows");
}


// ============================================================
// SINGLE COW
// ============================================================

export async function getCow(cowId) {
  return request(`/cow/${cowId}`);
}


// ============================================================
// COW RISK
// ============================================================

export async function getCowRisk(cowId) {
  return request(`/cow/${cowId}/risk`);
}


// ============================================================
// COW FORECAST
// ============================================================

export async function getCowForecast(cowId) {
  return request(`/cow/${cowId}/forecast`);
}


// ============================================================
// HERD RISK
// ============================================================

export async function getHerdRisk() {
  return request("/herd-risk");
}


// ============================================================
// PRIORITY COWS
// ============================================================

export async function getPriorityCows() {
  return request("/herd-risk/priority");
}