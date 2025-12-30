/**
 * Login Page - User authentication.
 */

import React, { useState, useCallback, useEffect } from 'react';
import Layout from '@theme/Layout';
import { useHistory, useLocation } from '@docusaurus/router';
import BrowserOnly from '@docusaurus/BrowserOnly';
import Cookies from 'js-cookie';
import axios from 'axios';

const API_URL = 'https://humanoid-robotics-course-book-production.up.railway.app/api/v1';
const TOKEN_COOKIE = 'auth_token';

// Form styles (same as signup for consistency)
const styles = {
  container: {
    maxWidth: '400px',
    margin: '2rem auto',
    padding: '2rem',
    backgroundColor: 'var(--ifm-background-surface-color)',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
  },
  title: {
    textAlign: 'center' as const,
    marginBottom: '1.5rem',
    color: 'var(--ifm-heading-color)',
  },
  form: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1rem',
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.25rem',
  },
  label: {
    fontWeight: 600,
    color: 'var(--ifm-font-color-base)',
    fontSize: '0.9rem',
  },
  input: {
    padding: '0.75rem',
    border: '1px solid var(--ifm-color-emphasis-300)',
    borderRadius: '4px',
    fontSize: '1rem',
    backgroundColor: 'var(--ifm-background-color)',
    color: 'var(--ifm-font-color-base)',
  },
  inputError: {
    borderColor: 'var(--ifm-color-danger)',
  },
  errorText: {
    color: 'var(--ifm-color-danger)',
    fontSize: '0.8rem',
    marginTop: '0.25rem',
  },
  button: {
    padding: '0.75rem 1.5rem',
    backgroundColor: 'var(--ifm-color-primary)',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'pointer',
    marginTop: '0.5rem',
  },
  buttonDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  link: {
    textAlign: 'center' as const,
    marginTop: '1rem',
    color: 'var(--ifm-font-color-secondary)',
  },
  alert: {
    padding: '0.75rem',
    borderRadius: '4px',
    marginBottom: '1rem',
  },
  alertError: {
    backgroundColor: 'var(--ifm-color-danger-lightest)',
    color: 'var(--ifm-color-danger-darkest)',
    border: '1px solid var(--ifm-color-danger-light)',
  },
  alertSuccess: {
    backgroundColor: 'var(--ifm-color-success-lightest)',
    color: 'var(--ifm-color-success-darkest)',
    border: '1px solid var(--ifm-color-success-light)',
  },
};

function LoginForm(): JSX.Element {
  const history = useHistory();
  const location = useLocation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Check for registration success message
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (params.get('registered') === 'true') {
      setSuccessMessage('Account created successfully! Please log in.');
    }
  }, [location]);

  // Check if already logged in
  useEffect(() => {
    const token = Cookies.get(TOKEN_COOKIE);
    if (token) {
      // Verify token is still valid
      axios.get(`${API_URL}/me`, {
        headers: { Authorization: `Bearer ${token}` }
      }).then(() => {
        history.push('/');
      }).catch(() => {
        Cookies.remove(TOKEN_COOKIE);
      });
    }
  }, [history]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/login`, {
        email,
        password,
      });

      const { access_token } = response.data;

      // Save token to cookie (expires in 24 hours)
      Cookies.set(TOKEN_COOKIE, access_token, { expires: 1, sameSite: 'Lax' });

      // Redirect to home page
      history.push('/');

      // Force page reload to update navbar state
      window.location.href = '/';
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Login failed. Please check your credentials.';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Welcome Back</h1>

      {successMessage && (
        <div style={{ ...styles.alert, ...styles.alertSuccess }}>{successMessage}</div>
      )}

      {error && (
        <div style={{ ...styles.alert, ...styles.alertError }}>{error}</div>
      )}

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.formGroup}>
          <label htmlFor="email" style={styles.label}>Email</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              setError(null);
            }}
            style={styles.input}
            placeholder="you@example.com"
            autoComplete="email"
          />
        </div>

        <div style={styles.formGroup}>
          <label htmlFor="password" style={styles.label}>Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => {
              setPassword(e.target.value);
              setError(null);
            }}
            style={styles.input}
            placeholder="Enter your password"
            autoComplete="current-password"
          />
        </div>

        <button
          type="submit"
          disabled={isSubmitting || !email || !password}
          style={{
            ...styles.button,
            ...(isSubmitting || !email || !password ? styles.buttonDisabled : {}),
          }}
        >
          {isSubmitting ? 'Logging in...' : 'Log In'}
        </button>
      </form>

      <p style={styles.link}>
        Don't have an account? <a href="/signup">Sign up</a>
      </p>
    </div>
  );
}

export default function LoginPage(): JSX.Element {
  return (
    <Layout title="Log In" description="Log in to your account">
      <BrowserOnly fallback={<div style={{ textAlign: 'center', padding: '2rem' }}>Loading...</div>}>
        {() => <LoginForm />}
      </BrowserOnly>
    </Layout>
  );
}
