import "./Login.css";
import { useState } from "react";
import { loginUser } from "../services/api";
import {
  ShieldCheck,
  Lock,
  User,
  AlertCircle,
  Loader2
} from "lucide-react";

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();

    setError("");

    if (!username.trim() || !password.trim()) {
      setError("Please enter username and password.");
      return;
    }

    try {
      setLoading(true);

      const data = await loginUser(
        username.trim(),
        password
      );

      // Store authentication information
      localStorage.setItem("token", data.access_token);
      localStorage.setItem("username", data.username);
      localStorage.setItem("role", data.role);

      // Tell App.jsx that login was successful
      onLogin(data);

    } catch (err) {
      setError(
        err.message || "Login failed. Please check your credentials."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">

      <div className="login-card">

        {/* Logo */}
        <div className="login-logo">
          <ShieldCheck size={32} />
        </div>

        {/* Header */}
        <div className="login-header">
          <h1>Mastitis AI</h1>

          <p>
            Bovine Mastitis Risk Monitoring
          </p>
        </div>

        {/* Login Form */}
        <form
          className="login-form"
          onSubmit={handleSubmit}
        >

          {/* Username */}
          <div className="login-field">

            <label htmlFor="username">
              Username
            </label>

            <div className="login-input-wrapper">

              <User size={18} />

              <input
                id="username"
                type="text"
                placeholder="Enter your username"
                value={username}
                onChange={(event) =>
                  setUsername(event.target.value)
                }
                disabled={loading}
                autoComplete="username"
              />

            </div>

          </div>

          {/* Password */}
          <div className="login-field">

            <label htmlFor="password">
              Password
            </label>

            <div className="login-input-wrapper">

              <Lock size={18} />

              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(event) =>
                  setPassword(event.target.value)
                }
                disabled={loading}
                autoComplete="current-password"
              />

            </div>

          </div>

          {/* Error */}
          {error && (
            <div className="login-error">

              <AlertCircle size={17} />

              <span>
                {error}
              </span>

            </div>
          )}

          {/* Login Button */}
          <button
            type="submit"
            className="login-button"
            disabled={loading}
          >

            {loading ? (
              <>
                <Loader2
                  size={18}
                  className="spin"
                />

                Signing in...
              </>
            ) : (
              <>
                <ShieldCheck size={18} />

                Sign in
              </>
            )}

          </button>

        </form>

        {/* Footer */}
        <div className="login-footer">

          <span>
            Secure access to Mastitis AI
          </span>

        </div>

      </div>

    </div>
  );
}

export default Login;