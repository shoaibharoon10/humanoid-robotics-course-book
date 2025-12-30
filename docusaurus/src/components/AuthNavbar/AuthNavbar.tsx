/**
 * AuthNavbar - Dynamic navbar items for authentication.
 * Shows Login/Signup when logged out, shows username/Logout when logged in.
 */

import React, { useState, useEffect } from 'react';
import Cookies from 'js-cookie';
import axios from 'axios';

const API_URL = 'https://humanoid-robotics-course-book-production.up.railway.app/api/v1';
const TOKEN_COOKIE = 'auth_token';

interface User {
  id: string;
  email: string;
  username: string;
}

const styles = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  link: {
    padding: '0.4rem 0.8rem',
    color: 'var(--ifm-navbar-link-color)',
    textDecoration: 'none',
    fontWeight: 500,
    fontSize: '0.9rem',
    borderRadius: '4px',
    transition: 'background-color 0.2s',
  },
  primaryButton: {
    backgroundColor: 'var(--ifm-color-primary)',
    color: 'white',
  },
  username: {
    color: 'var(--ifm-navbar-link-color)',
    fontWeight: 500,
    fontSize: '0.9rem',
    marginRight: '0.5rem',
  },
  logoutButton: {
    padding: '0.4rem 0.8rem',
    backgroundColor: 'transparent',
    color: 'var(--ifm-navbar-link-color)',
    border: '1px solid var(--ifm-color-emphasis-300)',
    borderRadius: '4px',
    cursor: 'pointer',
    fontWeight: 500,
    fontSize: '0.9rem',
  },
};

export default function AuthNavbar(): JSX.Element {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      const token = Cookies.get(TOKEN_COOKIE);
      if (!token) {
        setIsLoading(false);
        return;
      }

      try {
        const response = await axios.get(`${API_URL}/me`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setUser(response.data);
      } catch {
        Cookies.remove(TOKEN_COOKIE);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  const handleLogout = () => {
    Cookies.remove(TOKEN_COOKIE);
    setUser(null);
    window.location.href = '/';
  };

  if (isLoading) {
    return <div style={styles.container} />;
  }

  if (user) {
    return (
      <div style={styles.container}>
        <span style={styles.username}>Hi, {user.username}</span>
        <button
          onClick={handleLogout}
          style={styles.logoutButton}
        >
          Logout
        </button>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <a href="/login" style={styles.link}>
        Log In
      </a>
      <a href="/signup" style={{ ...styles.link, ...styles.primaryButton }}>
        Sign Up
      </a>
    </div>
  );
}
