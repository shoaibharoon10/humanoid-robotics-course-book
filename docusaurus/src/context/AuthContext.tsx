/**
 * AuthContext - Manages user authentication state across the application.
 */

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import Cookies from 'js-cookie';
import axios from 'axios';

// API URL - defaults to production Railway URL
const API_URL = 'https://humanoid-robotics-course-book-production.up.railway.app/api/v1';

// Token cookie name
const TOKEN_COOKIE = 'auth_token';

// User interface matching backend UserResponse
interface User {
  id: string;
  email: string;
  username: string;
  phone_number?: string;
  created_at: string;
}

// Auth context interface
interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, username: string, password: string, phoneNumber?: string) => Promise<void>;
  logout: () => void;
  error: string | null;
  clearError: () => void;
}

// Create context with default values
const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,
  login: async () => {},
  signup: async () => {},
  logout: () => {},
  error: null,
  clearError: () => {},
});

// Auth Provider component
export function AuthProvider({ children }: { children: ReactNode }): JSX.Element {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Check for existing token on mount
  useEffect(() => {
    const savedToken = Cookies.get(TOKEN_COOKIE);
    if (savedToken) {
      setToken(savedToken);
      // Fetch user info
      fetchCurrentUser(savedToken);
    } else {
      setIsLoading(false);
    }
  }, []);

  // Fetch current user from API
  const fetchCurrentUser = async (authToken: string) => {
    try {
      const response = await axios.get(`${API_URL}/me`, {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
      });
      setUser(response.data);
      setToken(authToken);
    } catch (err) {
      // Token is invalid, clear it
      Cookies.remove(TOKEN_COOKIE);
      setToken(null);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Login function
  const login = useCallback(async (email: string, password: string) => {
    setError(null);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_URL}/login`, {
        email,
        password,
      });

      const { access_token } = response.data;

      // Save token to cookie (expires in 24 hours)
      Cookies.set(TOKEN_COOKIE, access_token, { expires: 1, sameSite: 'Lax' });
      setToken(access_token);

      // Fetch user info
      await fetchCurrentUser(access_token);
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Login failed. Please try again.';
      setError(message);
      throw new Error(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Signup function
  const signup = useCallback(async (
    email: string,
    username: string,
    password: string,
    phoneNumber?: string
  ) => {
    setError(null);
    setIsLoading(true);

    try {
      await axios.post(`${API_URL}/signup`, {
        email,
        username,
        password,
        phone_number: phoneNumber || null,
      });

      // After successful signup, auto-login
      await login(email, password);
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Signup failed. Please try again.';
      setError(message);
      throw new Error(message);
    } finally {
      setIsLoading(false);
    }
  }, [login]);

  // Logout function
  const logout = useCallback(() => {
    Cookies.remove(TOKEN_COOKIE);
    setToken(null);
    setUser(null);
    setError(null);
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const value: AuthContextType = {
    user,
    token,
    isAuthenticated: !!token && !!user,
    isLoading,
    login,
    signup,
    logout,
    error,
    clearError,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook to use auth context
export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default AuthContext;
