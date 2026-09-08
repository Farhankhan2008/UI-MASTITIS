const API_BASE_URL = "http://127.0.0.1:8000";

async function request(endpoint) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }

  return response.json();
}

export async function getHealth() {
  return request("/health");
}

export async function getCows() {
  return request("/cows");
}

export async function getCow(cowId) {
  return request(`/cow/${cowId}`);
}

export async function getCowRisk(cowId) {
  return request(`/cow/${cowId}/risk`);
}

export async function getCowForecast(cowId) {
  return request(`/cow/${cowId}/forecast`);
}

export async function getHerdRisk() {
  return request("/herd-risk");
}

export async function getPriorityCows() {
  return request("/herd-risk/priority");
}