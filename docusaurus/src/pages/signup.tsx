/**
 * Signup Page - User registration with real-time validation.
 */

import React, { useState, useEffect, useCallback } from 'react';
import Layout from '@theme/Layout';
import { useHistory } from '@docusaurus/router';
import BrowserOnly from '@docusaurus/BrowserOnly';

// Validation helpers
function validateEmail(email: string): string | null {
  if (!email) return 'Email is required';
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) return 'Please enter a valid email address';
  return null;
}

function validateUsername(username: string): string | null {
  if (!username) return 'Username is required';
  if (username.length < 3) return 'Username must be at least 3 characters';
  if (username.length > 50) return 'Username must be less than 50 characters';
  if (!/^[a-zA-Z0-9_]+$/.test(username)) return 'Username can only contain letters, numbers, and underscores';
  return null;
}

function validatePassword(password: string): string | null {
  if (!password) return 'Password is required';
  if (password.length < 8) return 'Password must be at least 8 characters';
  if (!/[a-zA-Z]/.test(password)) return 'Password must contain at least one letter';
  if (!/[0-9]/.test(password)) return 'Password must contain at least one number';
  return null;
}

function validateConfirmPassword(password: string, confirmPassword: string): string | null {
  if (!confirmPassword) return 'Please confirm your password';
  if (password !== confirmPassword) return 'Passwords do not match';
  return null;
}

function validatePhoneNumber(phone: string): string | null {
  if (!phone) return null; // Optional field
  if (phone.length > 20) return 'Phone number is too long';
  return null;
}

// Form styles
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
    backgroundColor: 'var(--ifm-color-danger-lightest)',
    color: 'var(--ifm-color-danger-darkest)',
    border: '1px solid var(--ifm-color-danger-light)',
  },
};

function SignupForm(): JSX.Element {
  const history = useHistory();
  const [formData, setFormData] = useState({
    email: '',
    username: '',
    phoneNumber: '',
    password: '',
    confirmPassword: '',
  });
  const [errors, setErrors] = useState<Record<string, string | null>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Import useAuth dynamically to avoid SSR issues
  const [auth, setAuth] = useState<any>(null);

  useEffect(() => {
    import('@site/src/context/AuthContext').then((module) => {
      // We can't call hooks here, so we'll use a different approach
    });
  }, []);

  // Real-time validation
  useEffect(() => {
    const newErrors: Record<string, string | null> = {};

    if (touched.email) {
      newErrors.email = validateEmail(formData.email);
    }
    if (touched.username) {
      newErrors.username = validateUsername(formData.username);
    }
    if (touched.phoneNumber) {
      newErrors.phoneNumber = validatePhoneNumber(formData.phoneNumber);
    }
    if (touched.password) {
      newErrors.password = validatePassword(formData.password);
    }
    if (touched.confirmPassword) {
      newErrors.confirmPassword = validateConfirmPassword(formData.password, formData.confirmPassword);
    }

    setErrors(newErrors);
  }, [formData, touched]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setSubmitError(null);
  }, []);

  const handleBlur = useCallback((e: React.FocusEvent<HTMLInputElement>) => {
    const { name } = e.target;
    setTouched((prev) => ({ ...prev, [name]: true }));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Mark all fields as touched
    setTouched({
      email: true,
      username: true,
      phoneNumber: true,
      password: true,
      confirmPassword: true,
    });

    // Validate all fields
    const emailError = validateEmail(formData.email);
    const usernameError = validateUsername(formData.username);
    const phoneError = validatePhoneNumber(formData.phoneNumber);
    const passwordError = validatePassword(formData.password);
    const confirmPasswordError = validateConfirmPassword(formData.password, formData.confirmPassword);

    if (emailError || usernameError || phoneError || passwordError || confirmPasswordError) {
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);

    try {
      // Use axios directly for signup
      const axios = (await import('axios')).default;
      const API_URL = 'https://humanoid-robotics-course-book-production.up.railway.app/api/v1';

      await axios.post(`${API_URL}/signup`, {
        email: formData.email,
        username: formData.username,
        password: formData.password,
        phone_number: formData.phoneNumber || null,
      });

      // Redirect to login page after successful signup
      history.push('/login?registered=true');
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Signup failed. Please try again.';
      setSubmitError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const isFormValid = !Object.values(errors).some((error) => error !== null) &&
    formData.email && formData.username && formData.password && formData.confirmPassword;

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Create Account</h1>

      {submitError && (
        <div style={styles.alert}>{submitError}</div>
      )}

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.formGroup}>
          <label htmlFor="email" style={styles.label}>Email *</label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            onBlur={handleBlur}
            style={{
              ...styles.input,
              ...(errors.email && touched.email ? styles.inputError : {}),
            }}
            placeholder="you@example.com"
          />
          {errors.email && touched.email && (
            <span style={styles.errorText}>{errors.email}</span>
          )}
        </div>

        <div style={styles.formGroup}>
          <label htmlFor="username" style={styles.label}>Username *</label>
          <input
            type="text"
            id="username"
            name="username"
            value={formData.username}
            onChange={handleChange}
            onBlur={handleBlur}
            style={{
              ...styles.input,
              ...(errors.username && touched.username ? styles.inputError : {}),
            }}
            placeholder="johndoe"
          />
          {errors.username && touched.username && (
            <span style={styles.errorText}>{errors.username}</span>
          )}
        </div>

        <div style={styles.formGroup}>
          <label htmlFor="phoneNumber" style={styles.label}>Phone Number (Optional)</label>
          <input
            type="tel"
            id="phoneNumber"
            name="phoneNumber"
            value={formData.phoneNumber}
            onChange={handleChange}
            onBlur={handleBlur}
            style={{
              ...styles.input,
              ...(errors.phoneNumber && touched.phoneNumber ? styles.inputError : {}),
            }}
            placeholder="+1 234 567 8900"
          />
          {errors.phoneNumber && touched.phoneNumber && (
            <span style={styles.errorText}>{errors.phoneNumber}</span>
          )}
        </div>

        <div style={styles.formGroup}>
          <label htmlFor="password" style={styles.label}>Password *</label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            onBlur={handleBlur}
            style={{
              ...styles.input,
              ...(errors.password && touched.password ? styles.inputError : {}),
            }}
            placeholder="Min 8 chars, letters & numbers"
          />
          {errors.password && touched.password && (
            <span style={styles.errorText}>{errors.password}</span>
          )}
        </div>

        <div style={styles.formGroup}>
          <label htmlFor="confirmPassword" style={styles.label}>Confirm Password *</label>
          <input
            type="password"
            id="confirmPassword"
            name="confirmPassword"
            value={formData.confirmPassword}
            onChange={handleChange}
            onBlur={handleBlur}
            style={{
              ...styles.input,
              ...(errors.confirmPassword && touched.confirmPassword ? styles.inputError : {}),
            }}
            placeholder="Re-enter your password"
          />
          {errors.confirmPassword && touched.confirmPassword && (
            <span style={styles.errorText}>{errors.confirmPassword}</span>
          )}
        </div>

        <button
          type="submit"
          disabled={isSubmitting || !isFormValid}
          style={{
            ...styles.button,
            ...(isSubmitting || !isFormValid ? styles.buttonDisabled : {}),
          }}
        >
          {isSubmitting ? 'Creating Account...' : 'Sign Up'}
        </button>
      </form>

      <p style={styles.link}>
        Already have an account? <a href="/login">Log in</a>
      </p>
    </div>
  );
}

export default function SignupPage(): JSX.Element {
  return (
    <Layout title="Sign Up" description="Create a new account">
      <BrowserOnly fallback={<div style={{ textAlign: 'center', padding: '2rem' }}>Loading...</div>}>
        {() => <SignupForm />}
      </BrowserOnly>
    </Layout>
  );
}
