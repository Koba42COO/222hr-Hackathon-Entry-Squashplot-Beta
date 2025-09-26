import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, throwError, of } from 'rxjs';
import { tap, catchError, map } from 'rxjs/operators';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';

import { environment } from '../../../environments/environment';

export interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  permissions: string[];
  avatar?: string;
  lastLogin?: Date;
  preferences?: {
    theme: 'light' | 'dark' | 'auto';
    notifications: boolean;
    language: string;
  };
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterData {
  email: string;
  password: string;
  name: string;
  username: string;
  confirmPassword: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  token_type: string;
  user: User;
}

export interface TokenPayload {
  sub: string;
  email: string;
  name: string;
  role: string;
  permissions: string[];
  exp: number;
  iat: number;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  private isAuthenticatedSubject = new BehaviorSubject<boolean>(false);
  public isAuthenticated$ = this.isAuthenticatedSubject.asObservable();

  private tokenRefreshTimer: any;

  constructor(
    private http: HttpClient,
    private router: Router
  ) {
    this.initializeAuth();
  }

  /**
   * Initialize authentication state from stored tokens
   */
  private initializeAuth(): void {
    const token = this.getStoredToken();
    if (token && !this.isTokenExpired(token)) {
      const user = this.getUserFromToken(token);
      if (user) {
        this.setCurrentUser(user);
        this.scheduleTokenRefresh();
      }
    } else {
      this.clearAuthData();
    }
  }

  /**
   * Login with email and password
   */
  login(credentials: LoginCredentials): Observable<AuthResponse> {
    console.log('🔐 Attempting login for:', credentials.email);

    return this.http.post<AuthResponse>(`${environment.apiUrl}/auth/login`, {
      username: credentials.email, // Backend expects 'username' field
      password: credentials.password
    }).pipe(
      tap(response => {
        console.log('✅ Login successful:', response);
        this.handleAuthSuccess(response);
      }),
      catchError(error => {
        console.error('❌ Login failed:', error);
        return this.handleAuthError(error);
      })
    );
  }

  /**
   * Register new user
   */
  register(data: RegisterData): Observable<AuthResponse> {
    console.log('📝 Attempting registration for:', data.email);

    if (data.password !== data.confirmPassword) {
      return throwError(() => new Error('Passwords do not match'));
    }

    return this.http.post<AuthResponse>(`${environment.apiUrl}/auth/register`, {
      email: data.email,
      password: data.password,
      name: data.name
    }).pipe(
      tap(response => {
        console.log('✅ Registration successful:', response);
        this.handleAuthSuccess(response);
      }),
      catchError(error => {
        console.error('❌ Registration failed:', error);
        return this.handleAuthError(error);
      })
    );
  }

  /**
   * Logout user
   */
  logout(): void {
    console.log('👋 Logging out user');
    
    // Clear token refresh timer
    if (this.tokenRefreshTimer) {
      clearTimeout(this.tokenRefreshTimer);
    }

    // Optional: Call logout endpoint to invalidate token on server
    const token = this.getStoredToken();
    if (token) {
      this.http.post(`${environment.apiUrl}/auth/logout`, {}).subscribe({
        next: () => console.log('✅ Server logout successful'),
        error: (error) => console.warn('⚠️ Server logout failed:', error)
      });
    }

    this.clearAuthData();
    this.router.navigate(['/auth/login']);
  }

  /**
   * Refresh authentication token
   */
  refreshToken(): Observable<AuthResponse> {
    const refreshToken = localStorage.getItem('refresh_token');
    
    if (!refreshToken) {
      return throwError(() => new Error('No refresh token available'));
    }

    return this.http.post<AuthResponse>(`${environment.apiUrl}/auth/refresh`, {
      refresh_token: refreshToken
    }).pipe(
      tap(response => {
        console.log('🔄 Token refresh successful');
        this.handleAuthSuccess(response);
      }),
      catchError(error => {
        console.error('❌ Token refresh failed:', error);
        this.logout();
        return throwError(() => error);
      })
    );
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    const token = this.getStoredToken();
    return token !== null && !this.isTokenExpired(token);
  }

  /**
   * Get current user
   */
  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }

  /**
   * Check if user has specific permission
   */
  hasPermission(permission: string): boolean {
    const user = this.getCurrentUser();
    return user?.permissions?.includes(permission) || false;
  }

  /**
   * Check if user has specific role
   */
  hasRole(role: string): boolean {
    const user = this.getCurrentUser();
    return user?.role === role;
  }

  /**
   * Update user profile
   */
  updateProfile(updates: Partial<User>): Observable<User> {
    return this.http.put<User>(`${environment.apiUrl}/auth/profile`, updates).pipe(
      tap(updatedUser => {
        this.setCurrentUser(updatedUser);
        console.log('✅ Profile updated:', updatedUser);
      }),
      catchError(error => {
        console.error('❌ Profile update failed:', error);
        return throwError(() => error);
      })
    );
  }

  /**
   * Change password
   */
  changePassword(currentPassword: string, newPassword: string): Observable<any> {
    return this.http.post(`${environment.apiUrl}/auth/change-password`, {
      current_password: currentPassword,
      new_password: newPassword
    }).pipe(
      tap(() => {
        console.log('✅ Password changed successfully');
      }),
      catchError(error => {
        console.error('❌ Password change failed:', error);
        return throwError(() => error);
      })
    );
  }

  /**
   * Request password reset
   */
  requestPasswordReset(email: string): Observable<any> {
    return this.http.post(`${environment.apiUrl}/auth/forgot-password`, { email }).pipe(
      tap(() => {
        console.log('✅ Password reset requested for:', email);
      }),
      catchError(error => {
        console.error('❌ Password reset request failed:', error);
        return throwError(() => error);
      })
    );
  }

  /**
   * Reset password with token
   */
  resetPassword(token: string, newPassword: string): Observable<any> {
    return this.http.post(`${environment.apiUrl}/auth/reset-password`, {
      token,
      new_password: newPassword
    }).pipe(
      tap(() => {
        console.log('✅ Password reset successful');
      }),
      catchError(error => {
        console.error('❌ Password reset failed:', error);
        return throwError(() => error);
      })
    );
  }

  // Private helper methods

  private handleAuthSuccess(response: AuthResponse): void {
    // Store tokens
    localStorage.setItem('access_token', response.access_token);
    localStorage.setItem('refresh_token', response.refresh_token);
    
    // Set current user
    this.setCurrentUser(response.user);
    
    // Schedule token refresh
    this.scheduleTokenRefresh(response.expires_in);
    
    console.log('🎉 Authentication successful for:', response.user.email);
  }

  private handleAuthError(error: any): Observable<never> {
    let errorMessage = 'Authentication failed';
    
    if (error.error?.message) {
      errorMessage = error.error.message;
    } else if (error.message) {
      errorMessage = error.message;
    }

    return throwError(() => new Error(errorMessage));
  }

  private setCurrentUser(user: User): void {
    this.currentUserSubject.next(user);
    this.isAuthenticatedSubject.next(true);
  }

  private clearAuthData(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    this.currentUserSubject.next(null);
    this.isAuthenticatedSubject.next(false);
  }

  private getStoredToken(): string | null {
    return localStorage.getItem('access_token');
  }

  private isTokenExpired(token: string): boolean {
    try {
      const payload = this.getTokenPayload(token);
      const currentTime = Math.floor(Date.now() / 1000);
      return payload.exp < currentTime;
    } catch {
      return true;
    }
  }

  private getTokenPayload(token: string): TokenPayload {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  }

  private getUserFromToken(token: string): User | null {
    try {
      const payload = this.getTokenPayload(token);
      return {
        id: payload.sub,
        email: payload.email,
        name: payload.name,
        role: payload.role,
        permissions: payload.permissions || []
      };
    } catch {
      return null;
    }
  }

  private scheduleTokenRefresh(expiresIn?: number): void {
    if (this.tokenRefreshTimer) {
      clearTimeout(this.tokenRefreshTimer);
    }

    const token = this.getStoredToken();
    if (!token) return;

    let refreshTime: number;
    
    if (expiresIn) {
      // Refresh 5 minutes before expiration
      refreshTime = (expiresIn - 300) * 1000;
    } else {
      try {
        const payload = this.getTokenPayload(token);
        const currentTime = Math.floor(Date.now() / 1000);
        const timeUntilExpiry = payload.exp - currentTime;
        // Refresh 5 minutes before expiration
        refreshTime = (timeUntilExpiry - 300) * 1000;
      } catch {
        // If we can't decode the token, try refreshing in 30 minutes
        refreshTime = 30 * 60 * 1000;
      }
    }

    // Don't schedule refresh if token expires too soon
    if (refreshTime > 0) {
      this.tokenRefreshTimer = setTimeout(() => {
        console.log('🔄 Auto-refreshing token...');
        this.refreshToken().subscribe({
          error: () => {
            console.warn('⚠️ Auto token refresh failed, user will need to login again');
          }
        });
      }, refreshTime);
    }
  }

  // Demo/Development helpers
  
  /**
   * Create demo user for development
   */
  createDemoUser(): User {
    return {
      id: 'demo-user-001',
      email: 'demo@chaios-platform.com',
      name: 'Demo User',
      role: 'user',
      permissions: ['consciousness', 'quantum', 'mathematics', 'chat', 'analytics'],
      preferences: {
        theme: 'dark',
        notifications: true,
        language: 'en'
      },
      lastLogin: new Date()
    };
  }

  /**
   * Login as demo user (for development)
   */
  loginAsDemo(): void {
    const demoUser = this.createDemoUser();
    const demoToken = this.createDemoToken(demoUser);
    
    localStorage.setItem('access_token', demoToken);
    localStorage.setItem('refresh_token', 'demo-refresh-token');
    
    this.setCurrentUser(demoUser);
    console.log('🎭 Logged in as demo user');
  }

  private createDemoToken(user: User): string {
    // Create a fake JWT token for demo purposes
    const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
    const payload = btoa(JSON.stringify({
      sub: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
      permissions: user.permissions,
      exp: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now
      iat: Math.floor(Date.now() / 1000)
    }));
    const signature = btoa('demo-signature');
    
    return `${header}.${payload}.${signature}`;
  }
}
